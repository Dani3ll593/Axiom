"""Agent 1 — Searcher: descomposición de query + 5 APIs académicas + access check.

Flujo:
  1. Descompone la pregunta del usuario en 5 queries (una por API) con Qwen 7B
     + instructor para validación estructurada.
  2. Lanza fetches async paralelos contra PubMed, OpenAlex, arXiv, Crossref y
     Scielo — esta última con timeout dedicado más alto (4.2s p50 medido,
     ver `axiom_tech.md § searcher.py`).
  3. Deduplica por DOI (lowercased) o por (título, año) si no hay DOI.
  4. Enriquece cada paper con `check_access_async` (Unpaywall + OpenAlex +
     Crossref, regla 2-de-3) para poblar `is_open` y `pdf_url`.
  5. Devuelve `papers_found` para el Screener.

Decisiones de diseño:
  - Cliente `httpx.AsyncClient` y semáforos `asyncio.Semaphore` se crean
    perezosamente dentro del event loop activo (ver `axiom_tech.md` —
    declarar a nivel de módulo rompe LangGraph).
  - `instructor` se usa en lugar de `llm_router.route_task` porque la
    descomposición devuelve JSON estructurado y queremos validación
    Pydantic + auto-retry.
  - Errores no fatales se acumulan en `errors`; el grafo nunca se aborta
    por una API caída.
"""
from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from typing import Literal
from urllib.parse import quote_plus

import httpx
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.prompts import SEARCHER_PROMPT
from src.state import AxiomState
from src.tools.access_check import check_access_async

logger = logging.getLogger(__name__)

# --- Tunables ---
LLM_TIMEOUT_S = 120.0
SCIELO_TIMEOUT_S = 20.0          # Scielo p50 ~4.2s, dejar margen
DEFAULT_HTTP_TIMEOUT_S = 30.0
MAX_RESULTS_PER_API = 50         # Por API; total ≤ 250 papers antes de dedup
ACCESS_CHECK_CONCURRENCY = 16    # Para no saturar Unpaywall/OpenAlex/Crossref

# Rate limits por API (req/seg sostenidos)
API_LIMITS = {
    "pubmed":   5,
    "openalex": 10,
    "arxiv":    3,
    "scielo":   5,
    "crossref": 10,
}


# --- Schemas (espejo de searcher_prompt.txt § FORMAT) ---
class APIQueries(BaseModel):
    pubmed:   str
    openalex: str
    arxiv:    str
    scielo:   str
    crossref: str


class SearchDecomposition(BaseModel):
    decomposition_rationale: str
    queries: APIQueries
    expected_recall: Literal["low", "medium", "high"]
    notes: str = ""


# --- Singletons lazy (event loop-safe) ---
_http_client: httpx.AsyncClient | None = None
_semaphores: dict[str, asyncio.Semaphore] = {}
_llm_client = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(DEFAULT_HTTP_TIMEOUT_S, connect=10.0),
            follow_redirects=True,
            headers={
                "User-Agent": f"Axiom/1.0 (mailto:{settings.contact_email})",
            },
        )
    return _http_client


def _get_semaphore(name: str) -> asyncio.Semaphore:
    """Crea el semáforo en el primer await — dentro del event loop activo."""
    if name not in _semaphores:
        _semaphores[name] = asyncio.Semaphore(API_LIMITS[name])
    return _semaphores[name]


def _get_llm_client():
    global _llm_client
    if _llm_client is None:
        _llm_client = instructor.from_openai(
            AsyncOpenAI(
                base_url=settings.vllm_url_7b,
                api_key=settings.vllm_api_key or "EMPTY",
            ),
            mode=instructor.Mode.JSON,
        )
    return _llm_client


# --- Step 1: Decomposition ---
async def _decompose_query(question: str, prisma: dict) -> SearchDecomposition | None:
    """Pide al 7B una descomposición a 5 queries. None si falla."""
    client = _get_llm_client()
    user_msg = f"research_question: {question}\nprisma_criteria: {prisma}"

    try:
        return await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.model_7b_name,
                response_model=SearchDecomposition,
                messages=[
                    {"role": "system", "content": SEARCHER_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                max_retries=2,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                temperature=0.3,
                max_tokens=1024,
            ),
            timeout=LLM_TIMEOUT_S,
        )
    except (ValidationError, asyncio.TimeoutError):
        logger.error("searcher: decomposition timeout/validation", extra={"node": "searcher"})
        return None
    except Exception:
        logger.exception("searcher: decomposition failed", extra={"node": "searcher"})
        return None


# --- Step 2: Per-API fetchers ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _fetch_pubmed(query: str) -> list[dict]:
    """E-utilities: esearch (IDs) → efetch (XML con abstracts)."""
    if not query.strip():
        return []
    async with _get_semaphore("pubmed"):
        client = _get_http_client()
        params_base = {"db": "pubmed"}
        if settings.pubmed_api_key:
            params_base["api_key"] = settings.pubmed_api_key

        es = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={**params_base, "term": query, "retmode": "json",
                    "retmax": MAX_RESULTS_PER_API},
        )
        es.raise_for_status()
        ids = es.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        ef = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={**params_base, "id": ",".join(ids),
                    "rettype": "abstract", "retmode": "xml"},
        )
        ef.raise_for_status()

    return _parse_pubmed_xml(ef.text)


def _parse_pubmed_xml(xml_text: str) -> list[dict]:
    """Parser minimal de PubmedArticleSet → dicts."""
    papers: list[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.warning("pubmed: failed to parse XML")
        return []

    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID", default="")
        title = art.findtext(".//ArticleTitle", default="")
        # AbstractText puede venir en múltiples nodos (background/methods/...)
        abstract_parts = [
            (t.text or "") for t in art.findall(".//Abstract/AbstractText")
        ]
        abstract = " ".join(p for p in abstract_parts if p).strip()

        year = art.findtext(".//PubDate/Year", default="") or \
               art.findtext(".//PubDate/MedlineDate", default="")[:4]

        authors = []
        for au in art.findall(".//AuthorList/Author"):
            last = au.findtext("LastName", default="")
            fore = au.findtext("ForeName", default="")
            name = f"{fore} {last}".strip()
            if name:
                authors.append(name)

        doi = ""
        for aid in art.findall(".//ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = (aid.text or "").strip()
                break

        papers.append({
            "paper_id": f"pubmed:{pmid}",
            "source":   "pubmed",
            "doi":      doi,
            "title":    title,
            "authors":  authors,
            "year":     year,
            "abstract": abstract,
        })
    return papers


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _fetch_openalex(query: str) -> list[dict]:
    """OpenAlex /works. La API key es obligatoria desde Feb 2026."""
    if not query.strip():
        return []
    async with _get_semaphore("openalex"):
        client = _get_http_client()
        params = {"search": query, "per-page": MAX_RESULTS_PER_API}
        if settings.openalex_api_key:
            params["api_key"] = settings.openalex_api_key

        r = await client.get("https://api.openalex.org/works", params=params)
        r.raise_for_status()
        data = r.json()

    papers: list[dict] = []
    for w in data.get("results", []):
        # OpenAlex serializa el abstract como inverted index — hay que reconstruir
        abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))
        doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
        papers.append({
            "paper_id": f"openalex:{w.get('id', '').rsplit('/', 1)[-1]}",
            "source":   "openalex",
            "doi":      doi,
            "title":    w.get("title", "") or "",
            "authors":  [a.get("author", {}).get("display_name", "")
                         for a in w.get("authorships", [])],
            "year":     str(w.get("publication_year", "")),
            "abstract": abstract,
        })
    return papers


def _reconstruct_abstract(inv_idx: dict | None) -> str:
    """OpenAlex abstract_inverted_index → texto plano."""
    if not inv_idx:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv_idx.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _fetch_arxiv(query: str) -> list[dict]:
    """arXiv API (Atom feed). Requiere HTTPS — http:// devuelve 301."""
    if not query.strip():
        return []
    async with _get_semaphore("arxiv"):
        client = _get_http_client()
        r = await client.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": query, "max_results": MAX_RESULTS_PER_API},
        )
        r.raise_for_status()

    return _parse_arxiv_atom(r.text)


def _parse_arxiv_atom(atom_text: str) -> list[dict]:
    ns = {"a": "http://www.w3.org/2005/Atom"}
    papers: list[dict] = []
    try:
        root = ET.fromstring(atom_text)
    except ET.ParseError:
        logger.warning("arxiv: failed to parse Atom feed")
        return []

    for entry in root.findall("a:entry", ns):
        eid = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
        # arXiv IDs vienen como "http://arxiv.org/abs/2401.12345v1"
        arxiv_id = eid.rsplit("/", 1)[-1]
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
        published = (entry.findtext("a:published", default="", namespaces=ns) or "")
        year = published[:4] if published else ""
        authors = [
            (a.findtext("a:name", default="", namespaces=ns) or "").strip()
            for a in entry.findall("a:author", ns)
        ]
        papers.append({
            "paper_id": f"arxiv:{arxiv_id}",
            "source":   "arxiv",
            "doi":      "",  # arXiv no garantiza DOI
            "title":    title,
            "authors":  [a for a in authors if a],
            "year":     year,
            "abstract": summary,
        })
    return papers


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _fetch_crossref(query: str) -> list[dict]:
    """Crossref /works. Polite pool via User-Agent + mailto."""
    if not query.strip():
        return []
    async with _get_semaphore("crossref"):
        client = _get_http_client()
        r = await client.get(
            "https://api.crossref.org/works",
            params={"query": query, "rows": MAX_RESULTS_PER_API},
        )
        r.raise_for_status()
        data = r.json()

    papers: list[dict] = []
    for item in data.get("message", {}).get("items", []):
        doi = (item.get("DOI") or "").lower()
        title_list = item.get("title") or []
        abstract = item.get("abstract") or ""  # raro pero a veces viene
        # Crossref a veces envuelve el abstract en <jats:p>...</jats:p>
        if abstract.startswith("<"):
            abstract = ET.fromstring(abstract).itertext()
            abstract = " ".join(abstract).strip()

        date_parts = (item.get("issued", {}).get("date-parts") or [[]])[0]
        year = str(date_parts[0]) if date_parts else ""

        authors = [
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in item.get("author", []) or []
        ]
        papers.append({
            "paper_id": f"crossref:{doi}" if doi else f"crossref:{item.get('URL', '')}",
            "source":   "crossref",
            "doi":      doi,
            "title":    title_list[0] if title_list else "",
            "authors":  [a for a in authors if a],
            "year":     year,
            "abstract": abstract,
        })
    return papers


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _fetch_scielo(query: str) -> list[dict]:
    """Scielo articlemeta. NO incluir en check_access_async (timeout 10s).

    Endpoint público. Si la API cambia, la respuesta se inspecciona y se
    captura como error no fatal — no rompe el grafo.
    """
    if not query.strip():
        return []
    async with _get_semaphore("scielo"):
        client = _get_http_client()
        r = await client.get(
            "https://articlemeta.scielo.org/api/v1/article/",
            params={"q": query, "limit": MAX_RESULTS_PER_API},
        )
        r.raise_for_status()
        data = r.json()

    papers: list[dict] = []
    for obj in data.get("objects", []) or []:
        # articlemeta serializa metadata como SciELO PS — extracción defensiva
        pid = obj.get("code") or obj.get("publication_date", "")
        title = (obj.get("title", {}) or {}).get("_", "") if isinstance(obj.get("title"), dict) \
                else (obj.get("title") or "")
        authors = [
            f"{a.get('given_names', '')} {a.get('surname', '')}".strip()
            for a in obj.get("authors", []) or []
        ]
        abstract = ""
        # abstracts viene como dict por idioma: {"en": "...", "pt": "..."}
        abs_field = obj.get("abstracts") or obj.get("original_abstract") or {}
        if isinstance(abs_field, dict):
            abstract = abs_field.get("en") or next(iter(abs_field.values()), "")
        elif isinstance(abs_field, str):
            abstract = abs_field

        papers.append({
            "paper_id": f"scielo:{pid}",
            "source":   "scielo",
            "doi":      (obj.get("doi") or "").lower(),
            "title":    title,
            "authors":  [a for a in authors if a],
            "year":     str(obj.get("publication_year", "")),
            "abstract": abstract,
        })
    return papers


# --- Step 3: Dedup ---
def _dedupe(papers: list[dict]) -> list[dict]:
    """Dedupe por DOI lowercased; fallback a (title_norm, year)."""
    seen: set[str] = set()
    out: list[dict] = []
    for p in papers:
        doi = (p.get("doi") or "").lower().strip()
        if doi:
            key = f"doi:{doi}"
        else:
            title_norm = " ".join((p.get("title") or "").lower().split())
            key = f"tn:{title_norm}|{p.get('year', '')}"
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


# --- Step 4: Access enrichment ---
async def _enrich_with_access(papers: list[dict]) -> list[dict]:
    """Llama check_access_async en paralelo (capped) solo para papers con DOI."""
    sem = asyncio.Semaphore(ACCESS_CHECK_CONCURRENCY)

    async def _one(p: dict) -> dict:
        if not p.get("doi"):
            return {**p, "is_open": False, "pdf_url": None, "access_confidence": 0.0}
        async with sem:
            try:
                ac = await check_access_async(p["doi"])
                return {
                    **p,
                    "is_open":           ac.get("is_open", False),
                    "pdf_url":           ac.get("pdf_url"),
                    "access_confidence": ac.get("confidence", 0.0),
                }
            except Exception as e:
                logger.warning(
                    "searcher: access_check failed for %s: %s", p["doi"], e,
                )
                return {**p, "is_open": False, "pdf_url": None, "access_confidence": 0.0}

    return list(await asyncio.gather(*(_one(p) for p in papers)))


# --- LangGraph Node ---
async def run_searcher(state: AxiomState) -> dict:
    """Nodo del grafo. Devuelve `papers_found` y `errors` como deltas."""
    question = state.get("question") or state.get("query") or ""
    prisma = state.get("prisma_criteria") or {}

    if not question.strip():
        return {
            "errors": [{"node": "searcher", "error": "empty_question"}],
        }

    # 1. Decomposition
    decomp = await _decompose_query(question, prisma)
    if decomp is None:
        return {
            "errors": [{"node": "searcher", "error": "decomposition_failed"}],
        }

    q = decomp.queries
    logger.info(
        "searcher: descompuesto — recall=%s, notas=%r",
        decomp.expected_recall, decomp.notes or "",
    )

    # 2. Fetches paralelos. Scielo va aparte con timeout más amplio.
    other_tasks = [
        _fetch_pubmed(q.pubmed),
        _fetch_openalex(q.openalex),
        _fetch_arxiv(q.arxiv),
        _fetch_crossref(q.crossref),
    ]
    scielo_task = asyncio.wait_for(_fetch_scielo(q.scielo), timeout=SCIELO_TIMEOUT_S)

    results = await asyncio.gather(
        *other_tasks, scielo_task, return_exceptions=True,
    )

    sources = ["pubmed", "openalex", "arxiv", "crossref", "scielo"]
    all_papers: list[dict] = []
    errors: list[dict] = []
    for source, res in zip(sources, results):
        if isinstance(res, Exception):
            errors.append({
                "node":  "searcher",
                "source": source,
                "error": f"{type(res).__name__}: {res}",
            })
            continue
        all_papers.extend(res)

    # 3. Dedup
    deduped = _dedupe(all_papers)

    # 4. Access enrichment
    papers_found = await _enrich_with_access(deduped)

    n_open = sum(1 for p in papers_found if p.get("is_open"))
    logger.info(
        "searcher: %d papers (de %d crudos), %d open access, %d errores",
        len(papers_found), len(all_papers), n_open, len(errors),
    )

    return {
        "papers_found": papers_found,
        "errors":       errors,
    }