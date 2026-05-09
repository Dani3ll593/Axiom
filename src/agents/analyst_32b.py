"""Agent 4b — Analyst 32B (QwQ)."""

import asyncio
import json
import logging

from src.state import AxiomState
from src.config import settings
from src.tools.llm_router import LLM_32B, extract_json_from_response
from src.prompts import ANALYST_PROMPT_32B

logger = logging.getLogger(__name__)

# --- Tunables ---
MAX_CONCURRENT_CLUSTERS = 2   # más bajo: VRAM del 32B
TIMEOUT_S = 300.0             # margen amplio para razonamiento <think>


def _prune_extraction(paper: dict) -> dict:
    return {
        "paper_id":        paper.get("paper_id"),
        "study_design":    paper.get("study_design"),
        "methodology":     paper.get("methodology"),
        "variables":       paper.get("variables", []),    # PATCH #1
        "results":         paper.get("results"),
        "limitations":     paper.get("limitations"),
        "source_fragments": paper.get("source_fragments", {}),
    }


async def _analyze_cluster_32b(cluster: list[dict], index: int) -> dict | None:
    simplified_cluster = [_prune_extraction(p) for p in cluster]
    user_msg = f"INPUT CLUSTER:\n{json.dumps(simplified_cluster, ensure_ascii=False)}"

    try:
        response = await asyncio.wait_for(
            LLM_32B.chat.completions.create(
                model=settings.model_32b_name,
                messages=[
                    {"role": "system", "content": ANALYST_PROMPT_32B},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=4096,   # tokens extra para el bloque <think>
            ),
            timeout=TIMEOUT_S,
        )

        raw_text = response.choices[0].message.content
        parsed_json = extract_json_from_response(raw_text)

        clusters_out = parsed_json.get("synthesis_clusters", [])
        if not clusters_out:
            raise ValueError("No synthesis_clusters found in JSON")

        # PATCH #6: anclar al cluster origen.
        result = clusters_out[0]
        result["_cluster_index"] = index
        return result

    except Exception as e:
        logger.error(
            "analyst_32b falló en cluster %d: %s - %s",
            index, type(e).__name__, e,
            extra={"node": "analyst_32b", "cluster_index": index},
        )
        return None


async def analyst_32b_node(state: AxiomState) -> dict:
    clusters = state.get("clusters", [])
    if not clusters:
        logger.warning("analyst_32b: No clusters to process.")
        return {}

    sem = asyncio.Semaphore(MAX_CONCURRENT_CLUSTERS)

    async def _gated(cluster, idx):
        async with sem:
            return await _analyze_cluster_32b(cluster, idx)

    results = await asyncio.gather(
        *[_gated(c, i) for i, c in enumerate(clusters)],
        return_exceptions=True,
    )

    synthesis = []
    errors = []

    for i, res in enumerate(results):
        if isinstance(res, Exception):
            errors.append({"node": "analyst_32b", "cluster_index": i, "error": str(res)})
        elif res is None:
            errors.append({"node": "analyst_32b", "cluster_index": i, "error": "parse_timeout_error"})
        else:
            synthesis.append(res)

    return {"synthesis_32b": synthesis, "errors": errors}