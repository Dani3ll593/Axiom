from langgraph.graph import StateGraph, START, END
from typing import Literal

# 1. FIX: Importación correcta desde la raíz
from src.state import AxiomState

# 2. FIX: Importar los Agentes Reales que ya programamos
from src.agents.searcher import run_searcher
from src.agents.screener import screener_node
from src.agents.extractor import run_extractor

# ==============================================================================
# NODOS DUMMY (Stubs) - Pendientes de desarrollar en el futuro
# ==============================================================================
async def analyst_7b_node(state: AxiomState) -> dict:
    return {"synthesis_7b": []}

async def analyst_32b_node(state: AxiomState) -> dict:
    return {"synthesis_32b": []}

async def reconciler_node(state: AxiomState) -> dict:
    return {"consensus_clusters": []}

async def gapfinder_node(state: AxiomState) -> dict:
    return {"research_gaps": []}

async def writer_node(state: AxiomState) -> dict:
    return {"executive_report_md": "# Reporte Final\nEn construcción..."}

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
    builder.add_node("searcher", run_searcher)    # <-- Nuevo!
    builder.add_node("screener", screener_node)
    builder.add_node("extractor", run_extractor)  # <-- Reemplazado por el real
    builder.add_node("analyst_7b", analyst_7b_node)
    builder.add_node("analyst_32b", analyst_32b_node)
    builder.add_node("reconciler", reconciler_node)
    builder.add_node("gapfinder", gapfinder_node)
    builder.add_node("writer", writer_node)

    # 2. Definir Aristas (Flujo)
    builder.add_edge(START, "searcher")           # Empezamos en el Searcher
    builder.add_edge("searcher", "screener")      # Searcher -> Screener
    
    # Condicional post-screening (Salta al final si no hay papers)
    builder.add_conditional_edges("screener", check_screening_results)
    
    builder.add_edge("extractor", "analyst_7b")
    builder.add_edge("extractor", "analyst_32b")
    
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