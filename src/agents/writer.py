"""Agent 6 — Writer."""

import asyncio
import json
import logging
from collections import Counter, defaultdict

from pydantic import BaseModel, ValidationError

from src.state import AxiomState
from src.config import settings
from src.tools.llm_router import LLM_32B, extract_json_from_response
from src.prompts import WRITER_PROMPT, WRITER_APA7_RULES

logger = logging.getLogger(__name__)

# --- Tunables ---
TIMEOUT_S = 400.0
MAX_TOKENS = 6000


# --- Esquema de salida ---
class WriterOutput(BaseModel):
    executive_report_md:    str
    apa7_literature_review: str


# ============================================================================
# Helpers — Tabla de referencias APA 7 (short form)
# ============================================================================
def _last_name(author: str) -> str:
    """Heurística para extraer el apellido en formatos mixtos.

    Maneja:
      - "Last, First M"  → "Last"            (PubMed-style)
      - "First Last"     → "Last"            (OpenAlex/Crossref)
      - "First Middle Last" → "Last"
      - cadena vacía     → ""
    """
    author = (author or "").strip()
    if not author:
        return ""
    if "," in author:
        return author.split(",", 1)[0].strip()
    parts = author.split()
    return parts[-1] if parts else ""


def _normalize_year(year) -> str:
    """Year viene como int (2024), str ("2024"), "n.d." o "" — devolvemos siempre str."""
    if year is None:
        return "n.d."
    if isinstance(year, int):
        return str(year)
    s = str(year).strip()
    return s if s else "n.d."


def _short_citation(authors: list[str], year: str) -> str:
    """Genera la cita short-form APA 7 sin sufijo de desambiguación."""
    if not authors:
        return f"Anónimo, {year}"
    last_names = [_last_name(a) for a in authors if _last_name(a)]
    if not last_names:
        return f"Anónimo, {year}"
    if len(last_names) == 1:
        return f"{last_names[0]}, {year}"
    if len(last_names) == 2:
        return f"{last_names[0]} & {last_names[1]}, {year}"
    return f"{last_names[0]} et al., {year}"


def _build_references_table(papers: list[dict]) -> dict[str, str]:
    """{paper_id: 'Smith et al., 2023a'} con desambiguación a/b/c.

    Aplica la regla APA 7: si dos o más papers comparten (autores-cortos, año),
    se sufijan letras alfabéticamente para que cada cita sea única.
    """
    base: dict[str, tuple[str, str]] = {}  # pid -> (author_str, year_str)
    for p in papers:
        pid = p.get("paper_id")
        if not pid:
            continue
        authors = p.get("authors") or []
        year = _normalize_year(p.get("year"))
        # Citación base sin año todavía sufijado
        if not authors:
            author_str = "Anónimo"
        elif len(authors) == 1:
            author_str = _last_name(authors[0]) or "Anónimo"
        elif len(authors) == 2:
            author_str = f"{_last_name(authors[0])} & {_last_name(authors[1])}"
        else:
            author_str = f"{_last_name(authors[0])} et al."
        base[pid] = (author_str, year)

    # Agrupar por (autor_str, year) para desambiguar
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pid, key in base.items():
        groups[key].append(pid)

    table: dict[str, str] = {}
    for (author_str, year), pids in groups.items():
        if len(pids) == 1:
            table[pids[0]] = f"{author_str}, {year}"
        else:
            # Orden estable por paper_id para que las letras sean reproducibles
            for i, pid in enumerate(sorted(pids)):
                # a..z; si pasamos de 26 (improbable), seguimos con aa, ab...
                if i < 26:
                    suffix = chr(ord("a") + i)
                else:
                    suffix = chr(ord("a") + (i // 26) - 1) + chr(ord("a") + (i % 26))
                table[pid] = f"{author_str}, {year}{suffix}"
    return table


# ============================================================================
# Helpers — Restricted papers + PRISMA flow
# ============================================================================
def _build_restricted_list(screened_papers: list[dict]) -> list[dict]:
    """Solo los papers relevantes que NO son open access — para la sección
    'RESTRICTED ACCESS ARTICLES' del reporte ejecutivo."""
    restricted = []
    for p in screened_papers:
        if p.get("is_open"):
            continue
        restricted.append({
            "paper_id":         p.get("paper_id"),
            "title":            p.get("title"),
            "doi":              p.get("doi"),
            "source":           p.get("source"),
            "access_confidence": p.get("access_confidence"),
        })
    return restricted


def _build_prisma_flow(
    papers_found: list[dict],
    screened_papers: list[dict],
    papers_excluded: list[dict],
) -> dict:
    """Conteos para el PRISMA 2020 flow diagram."""
    excluded_by_reason = Counter()
    for p in papers_excluded:
        reason = (p.get("screening") or {}).get("reason") or "unspecified"
        excluded_by_reason[reason] += 1
    return {
        "found":              len(papers_found),
        "included":           len(screened_papers),
        "excluded_total":     len(papers_excluded),
        "excluded_by_reason": dict(excluded_by_reason),
    }


# ============================================================================
# LangGraph Node
# ============================================================================
async def run_writer(state: AxiomState) -> dict:
    """Genera el reporte ejecutivo y la sección APA7 a partir del estado."""
    papers_found    = state.get("papers_found", [])
    screened_papers = state.get("screened_papers", [])
    papers_excluded = state.get("papers_excluded", [])
    consensus       = state.get("consensus_clusters", [])
    gaps            = state.get("research_gaps", [])

    # --- Construir el payload completo que el prompt requiere ---
    references_table = _build_references_table(screened_papers)
    restricted_list  = _build_restricted_list(screened_papers)
    prisma_flow      = _build_prisma_flow(papers_found, screened_papers, papers_excluded)

    # Condensar consensos para ahorrar tokens
    consensus_findings = [
        {
            "claim":                c.get("core_claim"),
            "agreement_percentage": c.get("agreement_percentage"),
            "is_heterogeneous":     c.get("heterogeneity_detected"),
            "supporting_papers":    c.get("supporting_papers", []),
            "contradicting_papers": c.get("contradicting_papers", []),
            "neutral_papers":       c.get("neutral_papers", []),
            "contradictions_found": c.get("contradiction_quotes", {}),
        }
        for c in consensus
    ]

    payload = {
        "research_question": state.get("question", "Pregunta no definida"),
        "prisma_flow":       prisma_flow,
        "consensus_findings": consensus_findings,
        "verified_gaps":      gaps,
        "restricted_papers":  restricted_list,
        "references_table":   references_table,
    }

    # --- Sustituir el placeholder de reglas APA en el prompt ---
    system_prompt = WRITER_PROMPT.replace("{apa7_rules_text}", WRITER_APA7_RULES)

    user_msg = f"SYNTHESIS PAYLOAD:\n{json.dumps(payload, ensure_ascii=False)}"

    logger.info(
        "writer: %d incluidos, %d restringidos, %d gaps, %d clusters",
        prisma_flow["included"], len(restricted_list), len(gaps), len(consensus),
    )

    # --- Llamada al LLM ---
    try:
        response = await asyncio.wait_for(
            LLM_32B.chat.completions.create(
                model=settings.model_32b_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=MAX_TOKENS,
            ),
            timeout=TIMEOUT_S,
        )

        raw_text = response.choices[0].message.content
        parsed_json = extract_json_from_response(raw_text)
        validated = WriterOutput(**parsed_json)

        logger.info("writer: ¡Reporte generado con éxito!")
        return {
            "executive_report_md":    validated.executive_report_md,
            "apa7_literature_review": validated.apa7_literature_review,
        }

    except ValidationError as e:
        logger.error("writer: Pydantic validation error: %s", e)
        return {"errors": [{"node": "writer", "error": f"validation_error: {e}"}]}
    except Exception as e:
        logger.exception("writer: Fallo en la generación del reporte")
        return {"errors": [{"node": "writer", "error": str(e)}]}