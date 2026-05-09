"""Clusterer — agrupa extracciones por similitud semántica con BGE-M3."""

from __future__ import annotations
import json
import logging
from collections import Counter
from threading import Lock

import chromadb
from FlagEmbedding import BGEM3FlagModel
from sklearn.cluster import AgglomerativeClustering

from src.state import AxiomState
from src.config import settings

logger = logging.getLogger(__name__)

# --- Tunables ---
SINGLETON_RATIO_WARN = 0.70   # >50% singletons → emitir warning sugiriendo subir threshold

# --- Singletons ---
_bge_model: BGEM3FlagModel | None = None
_bge_lock = Lock()

_chroma_client: chromadb.PersistentClient | None = None


def get_bge_model() -> BGEM3FlagModel:
    """Singleton thread-safe para BGE-M3 en fp16."""
    global _bge_model
    if _bge_model is None:
        with _bge_lock:
            if _bge_model is None:
                logger.info("Cargando modelo BGE-M3 en memoria...")
                _bge_model = BGEM3FlagModel(
                    "BAAI/bge-m3",
                    use_fp16=True,   # ~50% menos VRAM vs fp32
                    device="cuda",
                )
    return _bge_model


def get_chroma_client() -> chromadb.PersistentClient:
    """Cliente persistente para la DB Vectorial."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _chroma_client


def _extraction_to_text(extraction: dict) -> str:
    """Convierte el JSON extraído en texto plano para el embedding semántico."""
    title       = extraction.get("title")       or ""
    results     = extraction.get("results")     or ""
    methodology = extraction.get("methodology") or ""
    return f"Title: {title}. Methodology: {methodology}. Results: {results}."


def _prune_for_estimate(p: dict) -> dict:
    """Réplica del pruning que hacen analyst_7b y analyst_32b antes de mandar al LLM.

    MUST stay in sync con `_prune_extraction` en src/agents/analyst_7b.py y
    src/agents/analyst_32b.py. Si esos cambian los campos, este también.
    """
    return {
        "paper_id":         p.get("paper_id"),
        "study_design":     p.get("study_design"),
        "methodology":      p.get("methodology"),
        "variables":        p.get("variables", []),
        "results":          p.get("results"),
        "limitations":      p.get("limitations"),
        "source_fragments": p.get("source_fragments", {}),
    }


def _enforce_size_budget(
    clusters: list[list[dict]], max_user_chars: int
) -> list[list[dict]]:
    """Parte cualquier cluster cuyo JSON pruned exceda el budget de chars.

    Garantiza que ningún cluster supere el context window del analyst más
    estricto (ver settings.analyst_max_user_chars). El split es greedy en orden
    original — papers consecutivos del cluster van al mismo sub-cluster hasta
    llenar el presupuesto. No hay re-clustering semántico recursivo.

    Edge case: si un único paper, ya pruned, excede el budget, se deja como
    sub-cluster de 1 (el analyst probablemente fallará en ese específico, pero
    el resto del pipeline sigue). Truncar contenido por paper requeriría otra
    capa fuera del scope de este fix.
    """
    out: list[list[dict]] = []
    splits_applied = 0

    for cluster in clusters:
        pruned = [_prune_for_estimate(p) for p in cluster]
        full_size = len(json.dumps(pruned, ensure_ascii=False))
        if full_size <= max_user_chars:
            out.append(cluster)
            continue

        # Excede: partir greedy
        sub: list[dict] = []
        sub_chars = 2  # corchetes []
        for paper, p_pruned in zip(cluster, pruned):
            paper_chars = len(json.dumps(p_pruned, ensure_ascii=False)) + 1  # coma
            # Cierra el sub actual si añadir este paper rebasa (excepto si está vacío,
            # para no dropear papers individuales gigantes).
            if sub and sub_chars + paper_chars > max_user_chars:
                out.append(sub)
                sub = []
                sub_chars = 2
            sub.append(paper)
            sub_chars += paper_chars
        if sub:
            out.append(sub)
        splits_applied += 1

    if splits_applied:
        logger.info(
            "clusterer: %d cluster(s) excedieron budget (%d chars) y fueron particionados. "
            "Total clusters tras split: %d.",
            splits_applied, max_user_chars, len(out),
        )
    return out


def _log_cluster_diagnostics(clusters: list[list[dict]], threshold: float) -> None:
    """Reporta el histograma de tamaños y warning si hay demasiados singletons."""
    if not clusters:
        return

    sizes = [len(c) for c in clusters]
    total_clusters = len(sizes)
    singletons = sum(1 for s in sizes if s == 1)
    singleton_ratio = singletons / total_clusters

    # Histograma compacto: cuántos clusters de cada tamaño
    size_dist = Counter(sizes)
    histogram = ", ".join(
        f"{size}p×{count}" for size, count in sorted(size_dist.items())
    )

    logger.info(
        "clusterer: %d clusters formados (threshold=%.2f) | distribución: %s | singletons: %d/%d (%.0f%%)",
        total_clusters, threshold, histogram, singletons, total_clusters, singleton_ratio * 100,
    )

    if singleton_ratio > SINGLETON_RATIO_WARN and total_clusters >= 4:
        logger.warning(
            "clusterer: %.0f%% de los clusters son singletons. El threshold actual (%.2f) "
            "podría ser demasiado estricto para este corpus. Considera subir "
            "settings.cluster_distance_threshold a 0.55-0.65 si los analistas reportan "
            "muchos clusters de 1 paper.",
            singleton_ratio * 100, threshold,
        )


def cluster_extractions(
    extractions: list[dict],
    distance_threshold: float | None = None,
) -> list[list[dict]]:
    """Genera embeddings y agrupa los papers por similitud semántica.

    Args:
        extractions: lista de extracciones del extractor.
        distance_threshold: opcional. Si es None, usa
            `settings.cluster_distance_threshold`. La métrica es coseno;
            threshold=0.35 es muy estricto (near-duplicates), 0.5 captura
            mismo subtopic, 0.65+ captura mismo dominio general.
    """
    if not extractions:
        return []
    if len(extractions) == 1:
        return [[extractions[0]]]

    threshold = (
        distance_threshold
        if distance_threshold is not None
        else settings.cluster_distance_threshold
    )

    model = get_bge_model()
    texts = [_extraction_to_text(e) for e in extractions]

    # 1. Embeddings densos
    embeddings = model.encode(texts, batch_size=16)["dense_vecs"]

    # 2. Clustering jerárquico por distancia coseno
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    # 3. Ensamblaje
    clusters_dict: dict[int, list[dict]] = {}
    for idx, label in enumerate(labels):
        clusters_dict.setdefault(int(label), []).append(extractions[idx])

    clusters = list(clusters_dict.values())

    # 4. Garantizar que ningún cluster excede el budget del analyst más estricto
    clusters = _enforce_size_budget(clusters, settings.analyst_max_user_chars)

    # 5. Diagnostics
    _log_cluster_diagnostics(clusters, threshold)

    return clusters


# --- LangGraph Node ---
def clusterer_node(state: AxiomState) -> dict:
    """Nodo del grafo para procesar los clusters."""
    extractions = state.get("extractions", [])

    if not extractions:
        logger.warning("clusterer: No hay extracciones para clusterizar.")
        return {"clusters": []}

    logger.info("clusterer: Iniciando agrupación de %d extracciones.", len(extractions))
    clusters = cluster_extractions(extractions)
    return {"clusters": clusters}