"""Agent 4r — Reconciler."""


import logging
from enum import Enum

from src.state import AxiomState

logger = logging.getLogger(__name__)


class ConsensusLevel(str, Enum):
    FULL           = "full_agreement"        # todos los papers: 7B y 32B coinciden
    PARTIAL        = "partial_agreement"     # alguna diferencia leve (uno marcó neutral)
    SPLIT          = "split_disagreement"    # contradicción directa (support vs contradict)
    UNVERIFIED_7B  = "unverified_7b_only"    # 32B falló este cluster
    UNVERIFIED_32B = "unverified_32b_only"   # 7B falló este cluster


# Regla de prevalencia ante discrepancia (32B canónico, sesgo escéptico)
PRIORITY_MAP = {
    "supporting_contradicting": "contradicting",
    "contradicting_supporting": "contradicting",
    "supporting_neutral":       "supporting",
    "neutral_supporting":       "supporting",
    "contradicting_neutral":    "contradicting",
    "neutral_contradicting":    "contradicting",
}


def _get_paper_verdict(cluster: dict, paper_id: str) -> str:
    """Extrae el veredicto normalizado de un paper dentro de un cluster."""
    for key in ("supporting_papers", "contradicting_papers", "neutral_papers"):
        if paper_id in cluster.get(key, []):
            return key.replace("_papers", "")
    return "neutral"


def _aggregate_consensus_level(details: dict) -> str:
    """Calcula el nivel agregado del cluster a partir de los niveles por paper.

    Reglas de promoción (peor caso domina):
      - Cualquier SPLIT      → cluster SPLIT
      - Cualquier PARTIAL    → cluster PARTIAL
      - Todos FULL           → cluster FULL
      - Sin papers           → FULL (no hay desacuerdo posible)
    """
    if not details:
        return ConsensusLevel.FULL.value
    levels = {d["level"] for d in details.values()}
    if ConsensusLevel.SPLIT.value in levels:
        return ConsensusLevel.SPLIT.value
    if ConsensusLevel.PARTIAL.value in levels:
        return ConsensusLevel.PARTIAL.value
    return ConsensusLevel.FULL.value


def _reconcile_cluster(c7b: dict, c32b: dict) -> dict:
    """Cruza los veredictos de ambos modelos para un cluster que ambos procesaron."""
    all_pids = set(
        c7b.get("supporting_papers", []) +
        c7b.get("contradicting_papers", []) +
        c7b.get("neutral_papers", [])
    ) | set(
        c32b.get("supporting_papers", []) +
        c32b.get("contradicting_papers", []) +
        c32b.get("neutral_papers", [])
    )

    reconciled = {"supporting": [], "contradicting": [], "neutral": []}
    details: dict[str, dict] = {}

    for pid in all_pids:
        v7b  = _get_paper_verdict(c7b,  pid)
        v32b = _get_paper_verdict(c32b, pid)

        if v7b == v32b:
            level = ConsensusLevel.FULL
            final = v7b
        else:
            key = f"{v7b}_{v32b}"
            final = PRIORITY_MAP.get(key, "neutral")
            level = (
                ConsensusLevel.SPLIT
                if {v7b, v32b} == {"supporting", "contradicting"}
                else ConsensusLevel.PARTIAL
            )

        details[pid] = {"v7b": v7b, "v32b": v32b, "final": final, "level": level.value}
        reconciled[final].append(pid)

    total = len(all_pids)
    agreement_pct = round(len(reconciled["supporting"]) / total * 100) if total > 0 else 0

    return {
        "core_claim": c32b.get("core_claim", c7b.get("core_claim")),
        "total_papers_in_cluster": total,
        "agreement_percentage":    agreement_pct,
        "supporting_papers":       reconciled["supporting"],
        "contradicting_papers":    reconciled["contradicting"],
        "neutral_papers":          reconciled["neutral"],
        "contradiction_quotes":    c32b.get("contradiction_quotes", {}),
        "temporal_trend":          c32b.get("temporal_trend"),
        "heterogeneity_detected":  len(reconciled["contradicting"]) > 0,
        "consensus_level":         _aggregate_consensus_level(details),
        "consensus_details":       details,
    }


def _build_fallback(cluster: dict, source_model: str) -> dict:
    """Mantiene la estructura cuando solo uno de los dos modelos respondió.

    `source_model` ∈ {"7b", "32b"} → determina el ConsensusLevel reportado.
    """
    sup = cluster.get("supporting_papers", [])
    con = cluster.get("contradicting_papers", [])
    neu = cluster.get("neutral_papers", [])
    all_pids = sup + con + neu

    if source_model == "7b":
        level = ConsensusLevel.UNVERIFIED_7B.value
        present_marker = {"v7b": "present", "v32b": "missing"}
    else:
        level = ConsensusLevel.UNVERIFIED_32B.value
        present_marker = {"v7b": "missing", "v32b": "present"}

    return {
        # Conservamos los campos canónicos del cluster fuente
        "core_claim":              cluster.get("core_claim"),
        "total_papers_in_cluster": len(all_pids),
        "agreement_percentage":    cluster.get("agreement_percentage", 0),
        "supporting_papers":       sup,
        "contradicting_papers":    con,
        "neutral_papers":          neu,
        "contradiction_quotes":    cluster.get("contradiction_quotes", {}),
        "temporal_trend":          cluster.get("temporal_trend"),
        "heterogeneity_detected":  cluster.get("heterogeneity_detected", len(con) > 0),
        "consensus_level":         level,
        "consensus_details": {
            pid: {**present_marker, "final": _get_paper_verdict(cluster, pid), "level": "unverified"}
            for pid in all_pids
        },
    }


def reconciler_node(state: AxiomState) -> dict:
    """Nodo del grafo: alinea los resultados de los dos analistas por
    `_cluster_index` (índice canónico del clusterer)."""
    res_7b  = state.get("synthesis_7b",  [])
    res_32b = state.get("synthesis_32b", [])

    if not res_7b and not res_32b:
        logger.warning("reconciler: ambos analistas vacíos, nada que reconciliar.")
        return {"consensus_clusters": []}

    # Indexar por _cluster_index. Cualquier cluster sin el campo (no debería
    # ocurrir si los analistas están actualizados) es loggeado y descartado.
    def _index(synth: list[dict], label: str) -> dict[int, dict]:
        idx_map: dict[int, dict] = {}
        for c in synth:
            idx = c.get("_cluster_index")
            if idx is None:
                logger.warning("reconciler: cluster de %s sin _cluster_index, descartado.", label)
                continue
            if idx in idx_map:
                logger.warning(
                    "reconciler: _cluster_index duplicado (%s) en %s, conservando el primero.",
                    idx, label,
                )
                continue
            idx_map[idx] = c
        return idx_map

    index_7b  = _index(res_7b,  "synthesis_7b")
    index_32b = _index(res_32b, "synthesis_32b")

    # Universo total = unión de los índices observados (cubre el caso "solo 32B")
    all_indices = sorted(set(index_7b) | set(index_32b))

    consensus = []
    full_count = partial_count = split_count = unverified_count = 0

    for idx in all_indices:
        c7b  = index_7b.get(idx)
        c32b = index_32b.get(idx)

        if c7b and c32b:
            cluster = _reconcile_cluster(c7b, c32b)
            level = cluster["consensus_level"]
            if level == ConsensusLevel.FULL.value:
                full_count += 1
            elif level == ConsensusLevel.SPLIT.value:
                split_count += 1
            else:
                partial_count += 1
        elif c7b:
            cluster = _build_fallback(c7b, source_model="7b")
            unverified_count += 1
        else:  # c32b
            cluster = _build_fallback(c32b, source_model="32b")
            unverified_count += 1

        # `_cluster_index` es interno del reconciler — no contamina downstream.
        cluster.pop("_cluster_index", None)
        consensus.append(cluster)

    logger.info(
        "reconciler: %d clusters → full=%d, partial=%d, split=%d, unverified=%d",
        len(consensus), full_count, partial_count, split_count, unverified_count,
    )
    return {"consensus_clusters": consensus}