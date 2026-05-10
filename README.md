# Axiom Backend — Multi-Agent Systematic Review Orchestrator

> FastAPI + LangGraph orchestrator for the Axiom academic due-diligence pipeline.
> Receives a research question + PRISMA criteria, runs a 9-node agent graph over
> Qwen2.5-7B and QwQ-32B (vLLM/MI300X), and streams progress to the frontend
> via Server-Sent Events.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![ROCm 6.2+](https://img.shields.io/badge/ROCm-6.2+-red.svg)](https://rocm.docs.amd.com/)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

---

## ⚡ TL;DR

```
INPUT     → AxiomState (research question + PRISMA criteria)
TRANSPORT → FastAPI · POST /pipeline/start · SSE /pipeline/stream/{run_id}
COMPUTE   → LangGraph · 9 nodes · 5 academic APIs · BGE-M3 + ChromaDB
MODELS    → Qwen2.5-7B-Instruct (port 8000) + QwQ-32B-Preview (port 8001)
HARDWARE  → AMD MI300X (vLLM, ROCm 6.2+)
OUTPUT    → executive_report.md + apa7_literature_review.md + 2 PDFs
```

This repo is the **backend only**. The Streamlit frontend lives in a separate
repo on Hugging Face Spaces and only knows the HTTP contract documented below.

---

## 📐 Architecture

```
                    ┌──────────────────────────────┐
                    │  FastAPI (axiom_api.py)      │
                    │  · sequential job queue      │
                    │  · per-run events.jsonl      │
                    │  · SSE stream + replay       │
                    └────────────┬─────────────────┘
                                 │ pipeline.astream()
                                 ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                   LangGraph (graph.py)                      │
   │                                                             │
   │  searcher ──► screener ──┬─► extractor ──► clusterer        │
   │                          │                       │          │
   │                          │                       ├─ analyst_7b ─┐
   │                          │                       │              ▼
   │                          │                       └─ analyst_32b ─► reconciler
   │                          │                                              │
   │                          │                                              ▼
   │                          └────────────────────────► writer ◄── gap_finder
   │                                                       │
   │                                                       ▼
   │                                                      END
   └─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │  vLLM × 2 (MI300X)                   │
              │  · Qwen2.5-7B-Instruct  → :8000      │
              │  · QwQ-32B-Preview      → :8001      │
              └──────────────────────────────────────┘
```

**Why two vLLM servers?** The 7B (fast cascade pass + JSON-restricted extraction
with `outlines`) and the 32B (chain-of-thought reasoning, narrative writing)
have very different latency profiles and concurrency budgets. Running them as
separate processes lets each tune `max_concurrent_*` independently.

**Why a sequential job queue?** A single MI300X has finite VRAM. Running two
pipelines in parallel risks OOMs. `axiom_api.py` accepts up to 10 queued runs
and processes them one-by-one (`MAX_QUEUE_SIZE`).

**Why per-run `events.jsonl`?** Durability — if the SSE consumer disconnects,
they can reconnect to `/pipeline/stream/{run_id}?since_event_id=N` and replay
everything from disk before subscribing to live events.

---

## 🤖 Agent Pipeline

| # | Node | Model | Role |
|---|---|---|---|
| 1 | `searcher` | Qwen2.5-7B (instructor) | 5 APIs in parallel · DOI dedupe · Unpaywall+OpenAlex+Crossref access check |
| 2 | `screener` | Qwen2.5-7B (LoRA) → QwQ-32B | PRISMA cascade — 7B fast pass, 32B re-evaluates `low confidence` / `uncertain` |
| 3 | `extractor` | Qwen2.5-7B (LoRA) | PyMuPDF → JSON schema (`outlines` for 0% malformed) |
| — | `clusterer` | BGE-M3 + AgglomerativeClustering | Internal tool — embeds extractions, clusters, persists in ChromaDB |
| 4a | `analyst_7b` | Qwen2.5-7B | Per-cluster consensus/contradiction analysis (parallel branch) |
| 4b | `analyst_32b` | QwQ-32B | Same task with deeper `<think>` reasoning (parallel branch) |
| — | `reconciler` | deterministic | Internal tool — merges 7B+32B verdicts (32B canonical, skeptical bias on disagreement) |
| 5 | `gap_finder` | QwQ-32B | 5 gap categories + secondary OpenAlex verification (rejects gaps already covered) |
| 6 | `writer` | QwQ-32B | Executive Markdown report + APA 7 narrative draft + WeasyPrint PDFs |

**Fan-out/fan-in.** After `clusterer`, both analysts run in parallel; `reconciler`
waits on both before continuing. LangGraph handles the join via the AxiomState
reducers (`Annotated[list, operator.add]` on `synthesis_7b` and `synthesis_32b`).

**Conditional skip.** If the screener rejects every paper, the graph routes
straight from `screener` → `writer` (skipping extractor → clusterer → analysts
→ gap_finder) so the user gets a "no eligible studies" report instead of a
crash. See `check_screening_results()` in `graph.py`.

---

## 📦 AxiomState

LangGraph passes a single TypedDict between nodes. Each node returns a dict of
*just the fields it modified*; LangGraph merges using the reducer declared per
field.

```python
class AxiomState(TypedDict, total=False):
    # Inputs
    sr_id: str
    domain: str
    question: str
    prisma_criteria: dict

    # Accumulators (operator.add — concurrent-safe)
    errors:          Annotated[list[dict], operator.add]
    papers_found:    Annotated[list[dict], operator.add]   # searcher
    screened_papers: Annotated[list[dict], operator.add]   # screener (include + uncertain)
    papers_excluded: Annotated[list[dict], operator.add]   # screener (audit trail)
    extractions:     Annotated[list[dict], operator.add]   # extractor
    synthesis_7b:    Annotated[list[dict], operator.add]   # analyst_7b
    synthesis_32b:   Annotated[list[dict], operator.add]   # analyst_32b

    # Atomic writes (single producer)
    clusters:           list[list[dict]]   # clusterer
    consensus_clusters: list[dict]         # reconciler
    research_gaps:      list[dict]         # gap_finder

    # Writer outputs
    executive_report_md:        str
    apa7_literature_review:     str
    executive_report_pdf_path:  str | None
    apa7_pdf_path:              str | None
```

Source: `state.py`. Errors never abort the graph — failed nodes append to
`errors` and downstream nodes degrade gracefully.

---

## 🔌 HTTP API

All endpoints (except `/healthz`) require `Authorization: Bearer ${AXIOM_BACKEND_API_KEY}`.

### `POST /pipeline/start`
Submit a run. Returns immediately; the actual pipeline executes when the
sequential worker picks it up.

**Request body:**
```json
{
  "sr_id": "abc12345",
  "domain": "education",
  "question": "What is the effect of mindfulness on academic burnout?",
  "prisma_criteria": {
    "framework": "PICOS",
    "prisma_version": "2020",
    "eligibility_criteria": { "...": "see prisma_criteria_template.json" },
    "screening_instructions": { "...": "..." }
  },
  "_frontend_meta": { "frontend_version": "1.0", "submitted_at": "ISO-8601" }
}
```

Required fields: `question`, `prisma_criteria`. Body capped at 100 KB. Queue
capped at 10 runs (`503` when full).

**Response (202):**
```json
{ "run_id": "abc12345", "status": "queued", "queue_position": 0, "created_at": "..." }
```

### `GET /pipeline/stream/{run_id}` (SSE)
Server-Sent Events stream. Replays all historical events from
`data/api_runs/{run_id}/events.jsonl`, then subscribes to live events until
`finished` or `error`.

**Query param:** `since_event_id` (int, default 0) — for resumable subscriptions.

**Event shape:**
```
event: agent_done
id: 7
data: {"type":"agent_done","event_id":7,"node":"screener","payload":{...},"level":"success","message":null,"timestamp":1715300000.0}
```

**Emitted types:**
- `agent_start` — once per run, with `node=null` and `payload={"sr_id":...}`
- `agent_done` — per node, `payload={ui_label, is_internal, stats}`
- `log` — captured from `logging.getLogger("src")` (info/warn/error from agents)
- `finished` — `payload={sr_id, duration_s}` on success, `payload={error}` on crash

Heartbeat: a `: keep-alive` SSE comment every 15s when no events flowing.

### `GET /pipeline/result/{run_id}`
Once `finished` has been emitted, returns the final state in the **adapted
frontend contract** (not the raw internal `AxiomState`):

```json
{
  "report_md":         "## 1. Estado del Campo\n...",
  "apa_draft":         "Las intervenciones basadas en mindfulness...",
  "gaps":              [{"category":"Poblacional","description":"..."}],
  "restricted_papers": [{"paper_id":"...","title":"...","doi":"...","journal":"...","oa_url":null}],
  "stats":             {"found":312,"included":87,"excluded":225,"restricted":14},
  "kappa":             null,
  "sr_id":                     "abc12345",
  "executive_report_pdf_path": "/data/results/Axiom_Report_abc12345.pdf",
  "apa7_pdf_path":             "/data/results/Axiom_APA7_abc12345.pdf"
}
```

The adapter (`_adapt_final_state_for_ui` in `axiom_api.py`) maps internal
field names — `executive_report_md` → `report_md`, `apa7_literature_review`
→ `apa_draft`, `research_gaps` → `gaps` — and derives `restricted_papers`
(filter `screened_papers` by `is_open=False`) and `stats` (counts from list
lengths). Keeps the frontend a thin client.

**Status codes:** `202` (still running), `404` (unknown run), `500` (run errored).

### `GET /pipeline/{run_id}/status`
Lightweight polling endpoint. Returns `status` (`queued|running|done|error`),
`queue_position`, `current_node`, `last_event_id`, and timestamps. Use this if
you can't hold an SSE connection open.

### `GET /pipeline/{run_id}/report.pdf` and `GET /pipeline/{run_id}/apa7.pdf`
Serve the two PDFs generated by `writer.py` via WeasyPrint. The PDFs are
written to `data/results/` during the run; these endpoints just hand back
`FileResponse`s.

**Status codes:** `200` (bytes), `202` (run still in progress), `404`
(no PDF — generation failed, or status not `done`).

### `GET /healthz`
No auth. Pings both vLLM servers (`localhost:8000` and `:8001`) and reports
queue depth. Returns `{ status, vllm_7b, vllm_32b, queue_depth }`.

---

## ⚙️ Configuration (`.env`)

Loaded by `config.py` via `pydantic-settings`. Unknown keys are ignored.

| Key | Required | Default | Use |
|---|---|---|---|
| `CONTACT_EMAIL` | ✅ | — | Polite-pool User-Agent for academic APIs (Crossref, OpenAlex, Unpaywall). Empty = startup error. |
| `AXIOM_BACKEND_API_KEY` | ✅ (env, not `.env`) | — | Bearer token clients must present. Read by `axiom_api.py` directly from `os.environ`. |
| `VLLM_URL_7B` | ✅ | `VLLM_URL_7B` | Base URL of the 7B vLLM (e.g. `http://localhost:8000/v1`). |
| `VLLM_URL_32B` | ✅ | `VLLM_URL_32B` | Base URL of the 32B vLLM (e.g. `http://localhost:8001/v1`). |
| `VLLM_API_KEY` | optional | `EMPTY` | Forwarded to the OpenAI client. `EMPTY` works for unauthenticated vLLM. |
| `MODEL_7B_NAME` | optional | `Qwen/Qwen2.5-7B-Instruct` | Sent in the `model` field of completion requests. |
| `MODEL_32B_NAME` | optional | `Qwen/QwQ-32B-Preview` | Same, for the 32B server. |
| `PUBMED_API_KEY` | optional | `None` | Lifts PubMed rate limits from 3 → 10 req/s. |
| `OPENALEX_API_KEY` | optional | `None` | Premium OpenAlex polite-pool. |
| `CHROMA_PERSIST_DIR` | optional | `./data/chroma_db` | Where ChromaDB writes per-run vector stores. |
| `CLUSTER_DISTANCE_THRESHOLD` | optional | `0.7` | Cosine *distance* (not similarity) for AgglomerativeClustering. See `config.py` for the calibration table. |
| `ANALYST_MAX_USER_CHARS` | optional | `16000` | Hard cap on the JSON payload sent to analysts. Prevents 7B context overflow. |

---

## 🚀 Deployment (MI300X droplet)

### 1 · System dependencies (WeasyPrint native libs)
```bash
sudo apt-get update && sudo apt-get install -y \
  libpango-1.0-0 libpangocairo-1.0-0 libcairo2 \
  libgdk-pixbuf2.0-0 shared-mime-info
```
Without these, `weasyprint` imports but PDF rendering silently fails (the
pipeline still completes; `writer.py` logs a warning and returns
`executive_report_pdf_path: None`).

### 2 · Python dependencies
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
ROCm 6.2+ is assumed on the host. vLLM is *not* in `requirements.txt` because
it has out-of-tree wheels for ROCm — install it from AMD's instructions
separately.

### 3 · Two vLLM servers (run on the host, outside the API venv)
```bash
# Terminal A
vllm serve Qwen/Qwen2.5-7B-Instruct  --port 8000 --host 0.0.0.0

# Terminal B
vllm serve Qwen/QwQ-32B-Preview      --port 8001 --host 0.0.0.0
```
Tune `--max-model-len`, `--gpu-memory-utilization`, etc. per your VRAM budget.
Verify both with `curl http://localhost:8000/v1/models` and `:8001`.

### 4 · `.env`
```ini
CONTACT_EMAIL=you@example.org
AXIOM_BACKEND_API_KEY=<random-bearer-token>
VLLM_URL_7B=http://localhost:8000/v1
VLLM_URL_32B=http://localhost:8001/v1
PUBMED_API_KEY=<optional>
OPENALEX_API_KEY=<optional>
```

### 5 · Start the API
```bash
uvicorn axiom_api:app --host 0.0.0.0 --port 7860 --workers 1
```
**`--workers 1` is mandatory.** The job queue, current-run tracker, and SSE
subscriber map are in-process state. Multiple workers would split the queue
across processes and break SSE replay.

### 6 · Smoke test
```bash
# Health
curl http://localhost:7860/healthz

# Submit
curl -X POST http://localhost:7860/pipeline/start \
  -H "Authorization: Bearer $AXIOM_BACKEND_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question":"...", "prisma_criteria":{...}}'

# Stream (replace <run_id>)
curl -N http://localhost:7860/pipeline/stream/<run_id> \
  -H "Authorization: Bearer $AXIOM_BACKEND_API_KEY"
```

---

## 📦 Project Structure

```
axiom-backend/
├── axiom_api.py             ◀── FastAPI · queue · SSE · final_state adapter
├── graph.py                 ◀── LangGraph DAG (build_axiom_graph)
├── state.py                 ◀── AxiomState TypedDict + reducers
├── config.py                ◀── pydantic-settings (.env loader)
├── requirements.txt
│
├── src/agents/
│   ├── searcher.py          ◀── 5 APIs · dedupe · access_check
│   ├── screener.py          ◀── 7B → 32B PRISMA cascade
│   ├── extractor.py         ◀── PDF parse + outlines/instructor JSON
│   ├── analyst_7b.py        ◀── per-cluster consensus (parallel)
│   ├── analyst_32b.py       ◀── per-cluster consensus with <think>
│   ├── gap_finder.py        ◀── 5 categories + OpenAlex verification
│   └── writer.py            ◀── narrative + WeasyPrint PDFs
│
├── src/tools/
│   ├── llm_router.py        ◀── 2 AsyncOpenAI clients + JSON extractor
│   ├── clusterer.py         ◀── BGE-M3 + ChromaDB + AgglomerativeClustering
│   ├── reconciler.py        ◀── deterministic 7B↔32B merge
│   ├── access_check.py      ◀── Unpaywall + OpenAlex + Crossref (2-of-3)
│   └── pdf_parser.py        ◀── PyMuPDF + abstract_only fallback
│
├── src/prompts/             ◀── *.txt prompts loaded at module import
│   ├── searcher_prompt.txt
│   ├── screener_prompt.txt
│   ├── screener_fewshot.txt
│   ├── extraction_prompt.txt
│   ├── analyst_prompt_7b.txt
│   ├── analyst_prompt_32b.txt
│   ├── gapfinder_prompt.txt
│   ├── writer_prompt.txt
│   └── writer_apa7_rules.txt
│
├── extractor_schema.json
├── prisma_criteria_template.json
│
└── data/                    ◀── created at runtime
    ├── api_runs/{run_id}/   ◀── initial_state.json · events.jsonl · meta.json · final_state.json
    ├── results/             ◀── Axiom_Report_*.pdf · Axiom_APA7_*.pdf
    └── chroma_db/           ◀── ChromaDB persistent store (BGE-M3 vectors)
```

---

## ⚠️ Known limitations

These are real gaps between the documented contract and current behavior. Worth
fixing eventually; flagged here so consumers don't get surprised.

- **`kappa` is never computed.** No node writes it to the state, so
  `/pipeline/result/{run_id}` always returns `kappa: null`. The frontend
  gracefully shows `—`. Adding it would mean computing inter-rater agreement
  between the 7B and 32B screening passes inside `screener.py`.

- **Granular SSE events are missing.** The frontend pipeline contract mentions
  `agent_progress`, `stat`, and `kappa` event types. The backend currently only
  emits `agent_start` (once, global), `agent_done` (per node), `log`, and
  `finished`. Live progress bars per agent stay at 0% in real mode (mock mode
  uses a richer script). To fix: emit `agent_progress` from inside long-running
  nodes (searcher fetches, screener cascade, writer generation).

- **Prompts as `.txt` files.** `from src.prompts import WRITER_PROMPT` etc.
  assumes `src/prompts/__init__.py` exposes each `.txt` as a module attribute
  via `Path(__file__).with_suffix('.txt').read_text()`. Make sure that
  `__init__.py` exists when restructuring.

- **WeasyPrint failures are silent (by design).** If the native libs are
  missing, the pipeline still completes — the PDFs are just `None` in the
  state, and `/pipeline/{run_id}/report.pdf` returns 404. Check
  `data/api_runs/{run_id}/events.jsonl` for warning logs.

---

## 📜 License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Credits

Built for the **AMD MI300X Hackathon** · May 2026
Stack: Qwen2.5 · QwQ-32B · BGE-M3 · vLLM · LangGraph · Unsloth · ChromaDB · FastAPI