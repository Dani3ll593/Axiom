"""Agent 4a — Analyst 7B."""


import asyncio
import json
import logging
 
from src.state import AxiomState
from src.config import settings
from src.tools.llm_router import LLM_7B, extract_json_from_response
from src.prompts import ANALYST_PROMPT_7B
 
logger = logging.getLogger(__name__)
 
# --- Tunables ---
MAX_CONCURRENT_CLUSTERS = 4
TIMEOUT_S = 180.0
 
 
def _prune_extraction(paper: dict) -> dict:
    """Filtra campos innecesarios para ahorrar tokens de contexto."""
    return {
        "paper_id":        paper.get("paper_id"),
        "study_design":    paper.get("study_design"),
        "methodology":     paper.get("methodology"),
        "variables":       paper.get("variables", []),    # PATCH #1
        "results":         paper.get("results"),
        "limitations":     paper.get("limitations"),
        "source_fragments": paper.get("source_fragments", {}),
    }
 
 
async def _analyze_cluster_7b(cluster: list[dict], index: int) -> dict | None:
    simplified_cluster = [_prune_extraction(p) for p in cluster]
    user_msg = f"INPUT CLUSTER:\n{json.dumps(simplified_cluster, ensure_ascii=False)}"
 
    try:
        response = await asyncio.wait_for(
            LLM_7B.chat.completions.create(
                model=settings.model_7b_name,
                messages=[
                    {"role": "system", "content": ANALYST_PROMPT_7B},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=2048,
            ),
            timeout=TIMEOUT_S,
        )
 
        raw_text = response.choices[0].message.content
        parsed_json = extract_json_from_response(raw_text)
 
        clusters_out = parsed_json.get("synthesis_clusters", [])
        if not clusters_out:
            raise ValueError("No synthesis_clusters found in JSON")
 
        result = clusters_out[0]
        result["_cluster_index"] = index
        return result
 
    except Exception as e:
        logger.error(
            "analyst_7b falló en cluster %d: %s - %s",
            index, type(e).__name__, e,
            extra={"node": "analyst_7b", "cluster_index": index},
        )
        return None
 
 
async def analyst_7b_node(state: AxiomState) -> dict:
    clusters = state.get("clusters", [])
    if not clusters:
        logger.warning("analyst_7b: No clusters to process.")
        return {}
 
    sem = asyncio.Semaphore(MAX_CONCURRENT_CLUSTERS)
 
    async def _gated(cluster, idx):
        async with sem:
            return await _analyze_cluster_7b(cluster, idx)
 
    results = await asyncio.gather(
        *[_gated(c, i) for i, c in enumerate(clusters)],
        return_exceptions=True,
    )
 
    synthesis = []
    errors = []
 
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            errors.append({"node": "analyst_7b", "cluster_index": i, "error": str(res)})
        elif res is None:
            errors.append({"node": "analyst_7b", "cluster_index": i, "error": "parse_timeout_error"})
        else:
            synthesis.append(res)
 
    return {"synthesis_7b": synthesis, "errors": errors}