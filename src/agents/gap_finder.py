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

        if count < 10:
            status = "confirmed"
            confidence = "High (No significant literature found)"
        elif count < 100:
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

    # 1. Inferencia con QwQ-32B
    try:
        logger.info("gapfinder: Solicitando propuesta de gaps al QwQ-32B...")
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
        validated = GapFinderOutput(**parsed_json)

    except ValidationError as e:
        logger.error("gapfinder: Pydantic validation error: %s", e)
        return {"research_gaps": [], "errors": [{"node": "gapfinder", "error": f"validation_error: {e}"}]}
    except Exception as e:
        logger.exception("gapfinder: LLM call failed")
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

    # 4. Filtrar rechazados (literatura abundante → no es vacío real)
    final_gaps = [g for g in verified_gaps if g["verification_status"] != "rejected"]

    logger.info(
        "gapfinder: %d gaps confirmados/parciales de %d propuestos.",
        len(final_gaps), len(verified_gaps),
    )

    return {"research_gaps": final_gaps, "errors": []}