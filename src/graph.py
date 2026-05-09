from langgraph.graph import StateGraph, START, END
from typing import Literal

# 1. FIX: Importación correcta desde la raíz
from src.state import AxiomState

# 2. FIX: Importar los Agentes Reales que ya programamos
from src.agents.searcher import run_searcher
from src.agents.screener import screener_node
from src.agents.extractor import run_extractor
from src.tools.clusterer import clusterer_node
from src.agents.analyst_7b import analyst_7b_node
from src.agents.analyst_32b import analyst_32b_node
from src.tools.reconciler import reconciler_node
from src.agents.gap_finder import run_gap_finder
from src.agents.writer import run_writer


# ==============================================================================
# LÓGICA DE CONDICIONES (Ruteo Dinámico)
# ==============================================================================
def check_screening_results(state: AxiomState) -> Literal["extractor", "writer"]:
    """Si el screener rechaza TODOS los papers, saltamos directo al writer para informar."""
    if not state.get("screened_papers"):
        return "writer"
    return "extractor"

# ==============================================================================
# CONSTRUCCIÓN DEL GRAFO
# ==============================================================================
def build_axiom_graph():
    builder = StateGraph(AxiomState)

    # 1. Agregar Nodos
    builder.add_node("searcher", run_searcher)    
    builder.add_node("screener", screener_node)
    builder.add_node("extractor", run_extractor)
    builder.add_node("clusterer", clusterer_node) 
    builder.add_node("analyst_7b", analyst_7b_node)
    builder.add_node("analyst_32b", analyst_32b_node)
    builder.add_node("reconciler", reconciler_node)
    builder.add_node("gapfinder", run_gap_finder)
    builder.add_node("writer", run_writer)

    # 2. Definir Aristas (Flujo)
    builder.add_edge(START, "searcher")           # Empezamos en el Searcher
    builder.add_edge("searcher", "screener")      # Searcher -> Screener
    
    # Condicional post-screening (Salta al final si no hay papers)
    builder.add_conditional_edges("screener", check_screening_results)
    
    # Extractor entrega resultados al clusterer
    builder.add_edge("extractor", "clusterer")
    
    # Fan-out: El clusterer alimenta en paralelo a ambos analistas    
    builder.add_edge("clusterer", "analyst_7b")
    builder.add_edge("clusterer", "analyst_32b")
    
    # FAN-IN: El reconciliador necesita que AMBOS analistas terminen
    builder.add_edge("analyst_7b", "reconciler")
    builder.add_edge("analyst_32b", "reconciler")
    
    builder.add_edge("reconciler", "gapfinder")
    builder.add_edge("gapfinder", "writer")
    builder.add_edge("writer", END)

    # Compilar y retornar
    return builder.compile()

# Exportamos el pipeline listo para usar
pipeline = build_axiom_graph()