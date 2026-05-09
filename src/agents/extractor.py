"""Agent 3 — Extractor de datos estructurados (Qwen 7B FT).

Recibe `screened_papers` del estado y utiliza generación restringida
(Instructor/JSON mode) para extraer variables metodológicas.

Antes de la llamada al LLM, los papers que traen `pdf_url` (poblado por el
Searcher vía access_check) se enriquecen con `full_text` descargando el PDF
y usando `pdf_parser.parse_pdf`. Si el parser cae a `abstract_only` (PDF
escaneado, descarga falla, etc.), el extractor cae al `abstract` —
comportamiento idéntico al previo.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Literal, Optional

import httpx
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError, field_validator

from src.config import settings
from src.prompts import EXTRACTION_PROMPT
from src.state import AxiomState
from src.tools.pdf_parser import parse_pdf

logger = logging.getLogger(__name__)

# --- Tunables ---
MAX_CONCURRENT = 16
LLM_TIMEOUT_S = 45.0
MAX_INPUT_CHARS = 8000  # Truncation limit to prevent 7B context overflow

# PDF fetch tunables — separados del LLM para no acoplar concurrencias
PDF_DOWNLOAD_CONCURRENCY = 8
PDF_DOWNLOAD_TIMEOUT_S = 30.0

# --- Schema Definitions ---
class VariableItem(BaseModel):
    name: str
    type: Literal[
        "independent",   # Variable manipulada o predictora
        "dependent",     # Variable de resultado (sinónimo de outcome en algunos campos)
        "outcome",       # Resultado clínico primario/secundario
        "covariate",     # Covariable ajustada en el modelo
        "control",       # Variable de control
        "mediator",      # Variable mediadora (explica el 'cómo' o 'por qué')
        "moderator",     # Variable moderadora (afecta la fuerza/dirección de la relación)
        "confounder",    # Variable de confusión identificada
        "predictor"      # Usado en modelos de regresión no causales
    ]
    measurement: Optional[str] = None

class SampleInfo(BaseModel):
    n: Optional[int] = None
    description: Optional[str] = None
    country: Optional[str] = None

class SourceFragments(BaseModel):
    title:        Optional[str] = None
    authors:      Optional[str] = None
    sample:       Optional[str] = None
    study_design: Optional[str] = None
    methodology:  Optional[str] = None
    results:      Optional[str] = None
    limitations:  Optional[str] = None

class PaperExtraction(BaseModel):
    # Prompts dictates EVIDENCE FIRST pattern
    source_fragments: SourceFragments
    
    paper_id:    Optional[str] = None
    doi:         Optional[str] = None
    title:       Optional[str] = None
    authors:     Optional[list[str]] = None 
    year:        int | str = "n.d."
    sample:      SampleInfo = SampleInfo() # Asumiendo que SampleInfo maneja sus propios opcionales
    study_design: Optional[str] = None
    methodology:  Optional[str] = None
    variables:   list[VariableItem] = [] 
    results:     Optional[str] = None
    limitations: Optional[str] = None

    @field_validator("year", mode="before")
    @classmethod
    def validate_year(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return "n.d."
        if isinstance(v, str) and v.isdigit():
            return int(v)  # normalizar a int siempre que sea posible
        return v

# --- LLM Client Setup ---
_clients: dict = {}

def _get_client(base_url: str):
    if base_url not in _clients:
        _clients[base_url] = instructor.from_openai(
            AsyncOpenAI(
                base_url=base_url,
                api_key=settings.vllm_api_key or "EMPTY",
            ),
            mode=instructor.Mode.JSON,
        )
    return _clients[base_url]

# --- PDF Fetch + Parse ---
_pdf_http_client: httpx.AsyncClient | None = None

def _get_pdf_http_client() -> httpx.AsyncClient:
    """Cliente httpx singleton para descargas de PDFs (lazy, event loop-safe)."""
    global _pdf_http_client
    if _pdf_http_client is None or _pdf_http_client.is_closed:
        _pdf_http_client = httpx.AsyncClient(
            timeout=PDF_DOWNLOAD_TIMEOUT_S,
            follow_redirects=True,
            headers={"User-Agent": f"Axiom/1.0 (mailto:{settings.contact_email})"},
        )
    return _pdf_http_client


async def _fetch_and_parse_pdf(pdf_url: str) -> dict:
    """Descarga PDF y delega a `parse_pdf`. Devuelve {strategy, text, error}."""
    try:
        client = _get_pdf_http_client()
        r = await client.get(pdf_url)
        r.raise_for_status()
    except Exception as e:
        return {
            "strategy": "abstract_only",
            "text":     None,
            "error":    f"download_failed: {type(e).__name__}",
        }
    return parse_pdf(r.content)


async def _enrich_with_pdf_text(papers: list[dict]) -> list[dict]:
    """Para cada paper con `pdf_url` y sin `full_text`, intenta descargar+parsear.

    Si `parse_pdf` cae a `abstract_only` (escaneado, fallo, etc.), el paper
    se devuelve sin `full_text` — `_extract_one` caerá al abstract.
    """
    sem = asyncio.Semaphore(PDF_DOWNLOAD_CONCURRENCY)

    async def _one(p: dict) -> dict:
        if (p.get("full_text") or "").strip():
            return p
        pdf_url = p.get("pdf_url")
        if not pdf_url:
            return p
        async with sem:
            result = await _fetch_and_parse_pdf(pdf_url)
        if result.get("strategy") == "pymupdf_full" and result.get("text"):
            return {**p, "full_text": result["text"]}
        logger.info(
            "extractor: PDF fallback to abstract for %s (reason=%s)",
            p.get("paper_id"), result.get("error") or result.get("strategy"),
        )
        return p

    return list(await asyncio.gather(*(_one(p) for p in papers)))

# --- Core Task ---
async def _extract_one(
    paper: dict,
    model: str,
    base_url: str,
) -> PaperExtraction | None:
    """Ejecuta la extracción sobre un único paper con reintentos."""
    client = _get_client(base_url)
    
    raw_text = paper.get("full_text") or paper.get("abstract") or ""
    text_to_process = raw_text[:MAX_INPUT_CHARS]

    try:
        return await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                response_model=PaperExtraction,
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user",   "content": f"Paper text:\n{text_to_process}"},
                ],
                max_retries=3,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                temperature=0.0,
            ),
            timeout=LLM_TIMEOUT_S,
        )

        if extraction:
            extraction.title = extraction.title or paper.get("title")
            extraction.authors = extraction.authors or paper.get("authors", [])
            extraction.doi = extraction.doi or paper.get("doi")
            if extraction.year in (None, "n.d.") and paper.get("year"):
                extraction.year = paper["year"]

    except (ValidationError, json.JSONDecodeError, asyncio.TimeoutError) as e:
        logger.error(
            f"Extraction parsing/timeout failed: {type(e).__name__}",
            extra={"paper_id": paper.get("paper_id"), "node": "extractor"}
        )
        return None
    except Exception as e:
        logger.exception(
            f"Extraction LLM call failed: {str(e)}",
            extra={"paper_id": paper.get("paper_id"), "model": model, "node": "extractor"},
        )
        return None

# --- LangGraph Node ---
async def run_extractor(state: AxiomState) -> dict:
    """Nodo del grafo. Extrae datos y devuelve deltas para los reducers."""
    papers = state.get("screened_papers", [])
    
    if not papers:
        logger.warning("extractor: screened_papers vacío", extra={"node": "extractor"})
        return {}

    # Enriquecer con full_text vía pdf_parser donde haya pdf_url disponible.
    # Los papers sin pdf_url o cuyo PDF no parsee bien caen al abstract (comportamiento previo).
    papers = await _enrich_with_pdf_text(papers)

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    valid_papers = []
    errors_delta: list[dict] = []

    # Short-circuit empty text before hitting the LLM API
    for p in papers:
        pid = p.get("paper_id")
        text = p.get("full_text") or p.get("abstract") or ""
        if not text.strip():
            errors_delta.append({
                "node": "extractor",
                "paper_id": pid,
                "error": "empty_paper_text"
            })
        else:
            valid_papers.append(p)

    async def _gated(p: dict):
        async with sem:
            return await _extract_one(
                paper=p, 
                model=settings.model_7b_name, 
                base_url=settings.vllm_url_7b
            )

    results = await asyncio.gather(
        *(_gated(p) for p in valid_papers),
        return_exceptions=True,
    )

    extractions_delta: list[dict] = []

    for paper, result in zip(valid_papers, results):
        pid = paper.get("paper_id")

        if isinstance(result, Exception):
            errors_delta.append({
                "node": "extractor",
                "paper_id": pid,
                "error": f"{type(result).__name__}: {result}",
            })
            continue

        if result is None:
            errors_delta.append({
                "node": "extractor",
                "paper_id": pid,
                "error": "extraction_failed_or_timed_out",
            })
            continue

        extraction_dict = result.model_dump()
        if not extraction_dict.get("paper_id"):
            extraction_dict["paper_id"] = pid
            
        extractions_delta.append(extraction_dict)
    
    return {
        "extractions": extractions_delta,
        "errors": errors_delta,
    }