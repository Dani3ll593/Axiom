"""Agent 5 — Gap Finder."""

import asyncio
import json
import logging

import httpx
from pydantic import BaseModel, ValidationError

from src.state import AxiomState
from src.config import settings
from src.tools.llm_router import LLM_32B, extract_json_from_response
from src.prompts import GAPFINDER_PROMPT

logger = logging.getLogger(__name__)

# --- Tunables ---
TIMEOUT_S = 300.0
OPENALEX_TIMEOUT_S = 15.0

# OpenAlex hit thresholds. Calibrados para keywords genéricos típicos del LLM
# (p. ej. "pediatric children" devuelve >100k hits → no es vacío real).
# Con queries de 3-5 términos, <50 indica laguna confirmada, <500 indica
# literatura emergente.
OA_CONFIRMED_MAX = 50
OA_PARTIAL_MAX   = 500

# Si los 5 gaps caen como `rejected`, rescatamos los N con menos hits
# como `partially_addressed` para que el reporte PRISMA Item 23d nunca
# salga vacío. Decisión de diseño: preferimos gaps débiles a no tener gaps.
RESCUE_TOP_N = 2

# --- Esquemas (espejo de gapfinder_prompt.txt § FORMAT) ---
class ProposedGap(BaseModel):
    description: str
    justification: str
    keywords: list[str] = []


class GapFinderOutput(BaseModel):
    """Output literal del prompt: 5 categorías nombradas."""
    population_gap:     ProposedGap
    methodological_gap: ProposedGap
    comparison_gap:     ProposedGap
    temporal_gap:       ProposedGap
    unanswered_question: ProposedGap


# --- Llamada al LLM con un reintento ---
async def _call_qwq_with_retry(user_msg: str) -> GapFinderOutput:
    """Llama a QwQ y, si la respuesta no parsea o no valida, reintenta una vez.

    QwQ-32B es errático: a veces devuelve prosa narrativa en vez de JSON
    (TEST_001), o un JSON con categorías faltantes (TEST_003). Un solo
    reintento suele bastar — la varianza del modelo es alta entre llamadas
    aunque el prompt sea idéntico.
    """
    last_err: Exception | None = None
    for attempt in (1, 2):
        try:
            response = await asyncio.wait_for(
                LLM_32B.chat.completions.create(
                    model=settings.model_32b_name,
                    messages=[
                        {"role": "system", "content": GAPFINDER_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0.4,
                    max_tokens=4096,
                ),
                timeout=TIMEOUT_S,
            )
            raw_text = response.choices[0].message.content
            parsed_json = extract_json_from_response(raw_text)
            return GapFinderOutput(**parsed_json)
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_err = e
            if attempt == 1:
                logger.warning(
                    "gapfinder: intento %d falló (%s); reintentando.",
                    attempt, type(e).__name__,
                )
            continue
    # Ambos intentos fallaron — propagar el último error
    raise last_err  # type: ignore[misc]


# --- Verificación en OpenAlex ---
async def _verify_gap_in_openalex(category: str, gap: ProposedGap) -> dict:
    """Confirma el vacío buscando en OpenAlex con los keywords propuestos."""
    # El prompt produce keywords; si vienen vacíos caemos a la descripción.
    query = " ".join(gap.keywords) if gap.keywords else gap.description

    params = {"search": query, "per-page": 1}
    if settings.openalex_api_key:
        params["api_key"] = settings.openalex_api_key

    base_payload = {
        "category":    category,
        "description": gap.description,
        "justification": gap.justification,
        "keywords":    gap.keywords,
        "verification_query": query,
    }

    try:
        async with httpx.AsyncClient(timeout=OPENALEX_TIMEOUT_S) as client:
            r = await client.get("https://api.openalex.org/works", params=params)
            r.raise_for_status()
            count = r.json().get("meta", {}).get("count", 0)

        if count < OA_CONFIRMED_MAX:
            status = "confirmed"
            confidence = "High (No significant literature found)"
        elif count < OA_PARTIAL_MAX:
            status = "partially_addressed"
            confidence = "Medium (Emerging literature exists)"
        else:
            status = "rejected"
            confidence = f"Low (Found {count} existing works)"

        return {
            **base_payload,
            "openalex_hits":       count,
            "verification_status": status,
            "confidence":          confidence,
        }

    except Exception as e:
        logger.warning("gapfinder: OpenAlex verification failed for %r: %s", query, e)
        return {
            **base_payload,
            "openalex_hits":       None,
            "verification_status": "unverified_api_error",
            "confidence":          "Unknown",
        }


def _rescue_gaps_if_all_rejected(verified_gaps: list[dict]) -> list[dict]:
    """Si los 5 gaps cayeron `rejected`, rescata los N con menos hits.

    Preferimos un reporte con gaps débiles a un reporte sin gaps.
    Si algunos `rejected` no tienen `openalex_hits` (API falló), no rescatamos
    esos — solo los que tienen un count numérico.
    """
    statuses = {g["verification_status"] for g in verified_gaps}
    if statuses != {"rejected"}:
        return verified_gaps  # Hay al menos un confirmed/partial → no rescate

    rescuable = [g for g in verified_gaps if g.get("openalex_hits") is not None]
    if not rescuable:
        return verified_gaps  # No podemos ordenar sin hits → respetar el rechazo

    # Top N con menos hits: los más probables de ser un vacío real
    rescuable.sort(key=lambda g: g["openalex_hits"])
    rescued_ids = {id(g) for g in rescuable[:RESCUE_TOP_N]}

    out = []
    for g in verified_gaps:
        if id(g) in rescued_ids:
            new = dict(g)
            new["verification_status"] = "partially_addressed"
            new["confidence"] = (
                f"Rescued (Lowest hit count among rejected gaps: {g['openalex_hits']})"
            )
            out.append(new)
        else:
            out.append(g)

    logger.info(
        "gapfinder: los 5 gaps fueron rejected. Rescatados %d con menor hit count.",
        len(rescued_ids),
    )
    return out


# --- LangGraph Node ---
async def run_gap_finder(state: AxiomState) -> dict:
    """Analiza consensos, propone 5 gaps por categoría y los verifica."""
    consensus_clusters = state.get("consensus_clusters", [])

    if not consensus_clusters:
        logger.warning("gapfinder: No consensus_clusters to analyze.")
        return {"research_gaps": [], "errors": [{"node": "gapfinder", "error": "empty_consensus"}]}

    # Payload condensado: solo lo que el prompt necesita ver
    summary = [
        {
            "core_claim":             c.get("core_claim"),
            "agreement_percentage":   c.get("agreement_percentage"),
            "heterogeneity_detected": c.get("heterogeneity_detected"),
            "contradictions":         c.get("contradiction_quotes", {}),
        }
        for c in consensus_clusters
    ]
    user_msg = f"CONSENSUS SUMMARY:\n{json.dumps(summary, ensure_ascii=False)}"

    # 1. Inferencia con QwQ-32B (con un reintento ante variabilidad del modelo)
    try:
        logger.info("gapfinder: Solicitando propuesta de gaps al QwQ-32B...")
        validated = await _call_qwq_with_retry(user_msg)
    except ValidationError as e:
        logger.error("gapfinder: Pydantic validation error tras reintento: %s", e)
        return {"research_gaps": [], "errors": [{"node": "gapfinder", "error": f"validation_error: {e}"}]}
    except Exception as e:
        logger.exception("gapfinder: LLM call failed tras reintento")
        return {"research_gaps": [], "errors": [{"node": "gapfinder", "error": str(e)}]}

    # 2. Mapear las 5 categorías a (category_label, ProposedGap)
    #    El category_label es el que después leerá el writer y el reporte PRISMA.
    gaps_to_verify = [
        ("population",          validated.population_gap),
        ("methodology",         validated.methodological_gap),
        ("comparison",          validated.comparison_gap),
        ("temporal",            validated.temporal_gap),
        ("unanswered_question", validated.unanswered_question),
    ]

    # 3. Verificación paralela en OpenAlex
    logger.info("gapfinder: Verificando 5 gaps propuestos en OpenAlex...")
    verified_gaps = await asyncio.gather(
        *(_verify_gap_in_openalex(cat, gap) for cat, gap in gaps_to_verify)
    )

    # 4. Rescate: si los 5 cayeron rejected, recuperar los más débiles
    verified_gaps = _rescue_gaps_if_all_rejected(list(verified_gaps))

    # 5. Filtrar rechazados (literatura abundante → no es vacío real)
    final_gaps = [g for g in verified_gaps if g["verification_status"] != "rejected"]

    logger.info(
        "gapfinder: %d gaps confirmados/parciales de %d propuestos.",
        len(final_gaps), len(verified_gaps),
    )

    return {"research_gaps": final_gaps, "errors": []}