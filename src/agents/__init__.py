"""
Módulo de agentes (Nodos de LangGraph) de Axiom.
Exponemos los nodos listos para ser importados por src/graph.py.
"""

from .screener import screener_node
from .extractor import run_extractor

# Agentes que sabemos que existen o están pendientes, comentados por seguridad:
# from .searcher import searcher_node  # (Asumiendo que así se llama tu nodo searcher)
# from .analyst_7b import analyst_7b_node
# from .analyst_32b import analyst_32b_node
# from .gap_finder import gapfinder_node
# from .writer import writer_node

__all__ = [
    "screener_node",
    "run_extractor",
]