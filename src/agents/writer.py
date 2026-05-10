"""Agent 6 — Writer."""

import asyncio
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import BaseModel, ValidationError

from src.state import AxiomState
from src.config import settings
from src.tools.llm_router import LLM_32B, extract_json_from_response
from src.prompts import WRITER_PROMPT, WRITER_APA7_RULES

logger = logging.getLogger(__name__)

# --- Tunables ---
TIMEOUT_S = 800.0
MAX_TOKENS = 8000

# Directorio donde se guardan los PDFs generados. axiom_api.py los lee desde aquí.
REPORTS_DIR = Path("data/results")


# --- Esquema de salida ---
class WriterOutput(BaseModel):
    executive_report_md:    str
    apa7_literature_review: str


# ============================================================================
# Helpers — Detección de idioma de la pregunta de investigación
# ============================================================================
# Stopwords muy frecuentes y específicas de cada idioma. La heurística no
# pretende ser un detector general — solo distingue ES vs EN, que es lo único
# que el corpus actual produce. Si en el futuro entran PT/FR, ampliar aquí o
# cambiar a `langdetect`.
_ES_MARKERS = {
    "el", "la", "los", "las", "de", "del", "en", "para", "por", "con", "sin",
    "que", "qué", "cuál", "cuáles", "cómo", "cuándo", "dónde", "es", "son",
    "y", "o", "u", "un", "una", "unos", "unas", "sobre", "entre",
    "efectividad", "eficacia", "comparado", "respecto",
}
_EN_MARKERS = {
    "the", "of", "in", "for", "with", "without", "and", "or", "an", "a",
    "what", "which", "how", "when", "where", "is", "are", "to", "from",
    "compared", "between", "among", "effectiveness", "efficacy",
}


def _detect_language(question: str) -> str:
    """Devuelve 'Spanish' o 'English' según el idioma de la pregunta.

    Heurística simple:
    1. Si hay caracteres acentuados típicos del español (á é í ó ú ñ ¿ ¡), es ES.
    2. Si no, cuenta tokens contra listas de stopwords ES vs EN; gana mayoría.
    3. Empate o sin señal → English (default conservador, igual al comportamiento
       actual del prompt antes de este cambio).
    """
    if not question or not question.strip():
        return "English"

    # 1. Acentos hispanos: señal fuerte
    if re.search(r"[áéíóúñÁÉÍÓÚÑ¿¡]", question):
        return "Spanish"

    # 2. Conteo de stopwords
    tokens = re.findall(r"\b[a-záéíóúñ]+\b", question.lower())
    es_hits = sum(1 for t in tokens if t in _ES_MARKERS)
    en_hits = sum(1 for t in tokens if t in _EN_MARKERS)

    if es_hits > en_hits:
        return "Spanish"
    return "English"


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
# Helpers — Llamada a QwQ-32B con un reintento (espejo del patrón en gap_finder)
# ============================================================================
async def _call_qwq_with_retry(system_prompt: str, user_msg: str) -> "WriterOutput":
    """Llama al writer y, si la respuesta no parsea o no valida, reintenta una vez.

    QwQ-32B es errático: a veces ignora las tags <json>...</json> y emite
    prosa narrativa ("Alright, I've got this task..."), o un JSON con campos
    faltantes. Un solo reintento con temperatura distinta suele bastar — la
    varianza del modelo es alta entre llamadas aunque el prompt sea idéntico.

    Mantiene parametros idénticos al call original en el primer intento; en el
    segundo sube ligeramente la temperatura para romper determinismos.
    """
    last_err: Exception | None = None
    for attempt, temperature in ((1, 0.3), (2, 0.55)):
        try:
            response = await asyncio.wait_for(
                LLM_32B.chat.completions.create(
                    model=settings.model_32b_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=temperature,
                    max_tokens=MAX_TOKENS,
                ),
                timeout=TIMEOUT_S,
            )
            raw_text = response.choices[0].message.content
            parsed_json = extract_json_from_response(raw_text)
            return WriterOutput(**parsed_json)
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_err = e
            if attempt == 1:
                logger.warning(
                    "writer: intento %d falló (%s); reintentando con temperature=0.55.",
                    attempt, type(e).__name__,
                )
            continue
    raise last_err  # type: ignore[misc]


# ============================================================================
# Helpers — Generación de PDFs (reporte ejecutivo + APA 7)
# ============================================================================
# Dos perfiles de styling: el reporte ejecutivo es markdown rico (headers,
# tablas, bullets); el APA 7 es prosa académica con sangría y referencias.
# Comparten base CSS pero divergen en algunos detalles.

# Base común: tipografía y márgenes A4
_CSS_BASE = """
@page { size: A4; margin: 22mm 25mm; }
body {
    font-family: 'DejaVu Sans', 'Helvetica Neue', 'Helvetica', 'Arial', sans-serif;
    color: #2a2a2a;
    line-height: 1.6;
    font-size: 11pt;
}
h1 {
    color: #1E3A8A;
    border-bottom: 2px solid #1E3A8A;
    padding-bottom: 6px;
    font-size: 20pt;
    margin-top: 0;
}
h2 {
    color: #2563EB;
    margin-top: 22px;
    font-size: 14pt;
    border-bottom: 1px solid #cbd5e0;
    padding-bottom: 4px;
}
h3 { color: #2563EB; font-size: 12pt; margin-top: 16px; }
.axiom-meta {
    background: #f8fafc;
    border-left: 3px solid #1E3A8A;
    padding: 10px 14px;
    margin: 16px 0 22px 0;
    font-size: 10pt;
    color: #475569;
}
.axiom-footer {
    margin-top: 28px; padding-top: 10px; border-top: 1px solid #cbd5e0;
    font-size: 8.5pt; color: #94a3b8; text-align: center;
}
"""

# Reporte ejecutivo: markdown rico con tablas, código, blockquotes
_CSS_EXECUTIVE = _CSS_BASE + """
p { margin: 8px 0; text-align: justify; }
ul, ol { margin: 8px 0 8px 18px; }
li { margin: 4px 0; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 10pt; }
th, td { border: 1px solid #cbd5e0; padding: 6px 10px; text-align: left; }
th { background: #f1f5f9; font-weight: 600; }
code {
    background: #f1f5f9; padding: 1px 5px; border-radius: 3px;
    font-family: 'Courier New', monospace; font-size: 9.5pt;
}
blockquote {
    border-left: 3px solid #2563EB; padding: 4px 14px;
    margin: 10px 0; color: #475569; font-style: italic;
}
"""

# APA 7: prosa académica, primera línea sangrada en párrafos, sin justify forzado
# (los reviewers APA prefieren left-aligned), referencias hanging indent
_CSS_APA7 = _CSS_BASE + """
body { font-family: 'DejaVu Serif', 'Times New Roman', 'Times', serif; }
p {
    margin: 0 0 6px 0;
    text-indent: 1.27cm;     /* 0.5 in — sangría APA 7 estándar */
    text-align: left;
}
p:first-of-type, h1 + p, h2 + p, h3 + p { text-indent: 0; }
h2 { text-align: center; text-transform: none; border: none; }
h3 { font-style: italic; border: none; }
ul, ol { margin: 6px 0 10px 24px; }
/* Referencias en hanging indent (al final del documento) */
.references p {
    text-indent: -1.27cm;
    padding-left: 1.27cm;
    margin-bottom: 8px;
}
em { font-style: italic; }
"""


def _render_pdf(
    markdown_text: str,
    output_path: Path,
    title: str,
    css: str,
    meta_block: str = "",
) -> Path | None:
    """Convierte Markdown a PDF en `output_path` con el CSS dado.

    LAZY IMPORT: WeasyPrint y markdown se cargan acá adentro para que el
    módulo siga importable aunque las libs nativas (libpango, libcairo) no
    estén en el sistema. Si fallan, devolvemos None y seguimos.
    """
    try:
        import markdown as md_lib
        from weasyprint import HTML
    except (ImportError, OSError) as e:
        logger.warning(
            "writer: WeasyPrint/markdown no disponibles (%s: %s). "
            "El PDF '%s' NO se generará.",
            type(e).__name__, e, output_path.name,
        )
        return None

    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        html_body = md_lib.markdown(
            markdown_text,
            extensions=["tables", "fenced_code", "nl2br"],
        )

        full_html = f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>{title}</title>
    <style>{css}</style>
  </head>
  <body>
    <h1>{title}</h1>
    {meta_block}
    {html_body}
    <div class="axiom-footer">
      Generated by Axiom · AMD MI300X · vLLM + LangGraph + QwQ-32B
    </div>
  </body>
</html>"""

        HTML(string=full_html).write_pdf(str(output_path))
        logger.info(
            "writer: PDF generado en %s (%.1f KB)",
            output_path, output_path.stat().st_size / 1024,
        )
        return output_path

    except Exception as e:
        logger.warning(
            "writer: falló la generación del PDF '%s' (%s: %s).",
            output_path.name, type(e).__name__, e,
        )
        return None


def _escape_html(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _generate_executive_pdf(markdown_text: str, sr_id: str, question: str) -> Path | None:
    """PDF del executive_report_md."""
    meta_block = (
        f'<div class="axiom-meta">'
        f'<strong>Run ID:</strong> {sr_id}<br/>'
        f'<strong>Research question:</strong> <em>{_escape_html(question)}</em>'
        f'</div>'
    )
    return _render_pdf(
        markdown_text=markdown_text,
        output_path=REPORTS_DIR / f"Axiom_Report_{sr_id}.pdf",
        title="Axiom — Systematic Review Report",
        css=_CSS_EXECUTIVE,
        meta_block=meta_block,
    )


def _generate_apa7_pdf(apa_text: str, sr_id: str, question: str) -> Path | None:
    """PDF del apa7_literature_review."""
    meta_block = (
        f'<div class="axiom-meta">'
        f'<strong>Run ID:</strong> {sr_id}<br/>'
        f'<strong>Research question:</strong> <em>{_escape_html(question)}</em>'
        f'</div>'
    )
    return _render_pdf(
        markdown_text=apa_text,
        output_path=REPORTS_DIR / f"Axiom_APA7_{sr_id}.pdf",
        title="Literature Review (APA 7)",
        css=_CSS_APA7,
        meta_block=meta_block,
    )


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
    question        = state.get("question", "Pregunta no definida")

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
        "research_question": question,
        "prisma_flow":       prisma_flow,
        "consensus_findings": consensus_findings,
        "verified_gaps":      gaps,
        "restricted_papers":  restricted_list,
        "references_table":   references_table,
    }

    # --- Sustituir placeholders del prompt: APA rules + idioma de salida ---
    output_language = _detect_language(question)
    logger.info("writer: idioma detectado para el reporte = %s", output_language)
    system_prompt = (
        WRITER_PROMPT
        .replace("{apa7_rules_text}", WRITER_APA7_RULES)
        .replace("{output_language}", output_language)
    )

    user_msg = f"SYNTHESIS PAYLOAD:\n{json.dumps(payload, ensure_ascii=False)}"

    logger.info(
        "writer: %d incluidos, %d restringidos, %d gaps, %d clusters",
        prisma_flow["included"], len(restricted_list), len(gaps), len(consensus),
    )

    # --- Llamada al LLM (con un reintento ante variabilidad de QwQ) ---
    try:
        validated = await _call_qwq_with_retry(system_prompt, user_msg)
        logger.info("writer: ¡Reporte generado con éxito!")

        # Generar los 2 PDFs (best-effort: si fallan, el run sigue siendo válido).
        sr_id = state.get("sr_id", "unknown")
        executive_pdf = _generate_executive_pdf(
            validated.executive_report_md, sr_id, question,
        )
        apa7_pdf = _generate_apa7_pdf(
            validated.apa7_literature_review, sr_id, question,
        )

        return {
            "executive_report_md":    validated.executive_report_md,
            "apa7_literature_review": validated.apa7_literature_review,
            # Paths (string) a cada PDF, o None si la generación falló.
            # axiom_api.py los lee para servir GET /pipeline/{run_id}/report.pdf
            # y /apa7.pdf (o el endpoint que decida exponer).
            "executive_report_pdf_path": str(executive_pdf) if executive_pdf else None,
            "apa7_pdf_path":             str(apa7_pdf) if apa7_pdf else None,
        }

    except ValidationError as e:
        logger.error("writer: Pydantic validation error tras reintento: %s", e)
        return {"errors": [{"node": "writer", "error": f"validation_error: {e}"}]}
    except Exception as e:
        logger.exception("writer: Fallo en la generación del reporte tras reintento")
        return {"errors": [{"node": "writer", "error": str(e)}]}