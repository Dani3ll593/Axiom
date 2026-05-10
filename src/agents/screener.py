"""Agent 2 — Screener PRISMA en cascada (Qwen 7B FT → QwQ-32B).
 
Recibe `papers_found` del searcher (arXiv, OpenAlex, Crossref, Scielo),
evalúa cada abstract contra los criterios PRISMA del estado y enruta cada
decisión a:
 
  - `screened_papers` ← decision == "include" OR "uncertain"
  - `papers_excluded` ← decision == "exclude"   (auditoría / PRISMA flow)
  - `errors`          ← fallos no fatales (timeout, red, validación)
 
Cascada
-------
El 7B fine-tuned hace el primer pase. Si su `confidence == "low"` o
`decision == "uncertain"`, se re-evalúa con QwQ-32B. El veredicto del 32B
es final; si el 32B falla, conservamos el del 7B (`route="7b_fallback"`)
en lugar de perder el paper.
 
Estrategia frente a la latencia de Scielo
-----------------------------------------
Scielo p50 ~4.2s (axiom_stack_completo2.md § Riesgo 5). El searcher la
aísla con timeout dedicado y `return_exceptions=True`, así que pueden
llegar papers con `abstract` vacío al screener. NO los mandamos al LLM
(gasto inútil de tokens y de la cuota de vLLM): short-circuit a
`decision="uncertain", reason="unavailable_full_text"` — valor permitido
por `prisma_criteria_template.json § exclusion_reasons_fixed_list`. El
paper queda en `screened_papers` y el writer lo reportará en la sección
de "acceso restringido".
"""
from __future__ import annotations

import asyncio
import re
import json
import ast
import logging
from typing import Literal, Optional

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from src.config import settings
from src.prompts import SCREENER_PROMPT, SCREENER_FEWSHOT
from src.state import AxiomState

logger = logging.getLogger(__name__)

# --- Tunables ---
MAX_CONCURRENT = 16  # Capped to prevent vLLM OOM on MI300X
LLM_TIMEOUT_S = 30.0
ESCALATE_CONFIDENCES = {"low"}
ESCALATE_DECISIONS = {"uncertain"}

# --- Schema Definitions ---
class CriteriaMet(BaseModel):
    population: Optional[bool] = False
    intervention: Optional[bool] = False
    outcomes: Optional[bool] = False
    study_design: Optional[bool] = False
    temporal: Optional[bool] = False
    language: Optional[bool] = False

class ScreenerDecision(BaseModel):
    chain_of_thought: Optional[str] = None
    justification: str
    criteria_met: CriteriaMet
    confidence: Literal["high", "medium", "low"]
    reason: Optional[str] = None
    decision: Literal["include", "exclude", "uncertain"]

# --- LLM Client Setup ---
_clients: dict = {}

def _get_client(base_url: str) -> instructor.AsyncInstructor:
    if base_url not in _clients:
        _clients[base_url] = instructor.from_openai(
            AsyncOpenAI(
                base_url=base_url,
                api_key=settings.vllm_api_key or "EMPTY",
            ),
            mode=instructor.Mode.JSON,
        )
    return _clients[base_url]

def limpiar_respuesta_qwq(texto_crudo: str) -> dict | None:
    """El extractor definitivo: Combina XML, Markdown, y Conteo de Llaves."""
    bloque_json = None
    
    # Intento 1: Buscar etiquetas XML <json>
    match_xml = re.search(r'<json>\s*(\{.*?\})\s*</json>', texto_crudo, re.DOTALL | re.IGNORECASE)
    if match_xml:
        bloque_json = match_xml.group(1)
    else:
        # Intento 2: Buscar bloque Markdown ```json
        match_md = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', texto_crudo, re.DOTALL | re.IGNORECASE)
        if match_md:
            bloque_json = match_md.group(1)
            
    # Intento 3: Búsqueda cruda de llaves a la fuerza
    if not bloque_json:
        match_llave = re.search(r'\{\s*["\'](chain_of_thought|justification)["\']', texto_crudo)
        inicio = match_llave.start() if match_llave else texto_crudo.find('{')
        if inicio != -1:
            sub_texto = texto_crudo[inicio:]
            llaves = 0
            for i, char in enumerate(sub_texto):
                if char == '{': llaves += 1
                elif char == '}':
                    llaves -= 1
                    if llaves == 0:
                        bloque_json = sub_texto[:i+1]
                        break

    if not bloque_json:
        return None

    # --- LAVADORA DE TEXTO ---
    bloque_json = bloque_json.replace('\xa0', ' ').replace('\t', ' ')
    bloque_json = re.sub(r'//.*', '', bloque_json) # Quita comentarios de una línea
    bloque_json = re.sub(r'/\*.*?\*/', '', bloque_json, flags=re.DOTALL) # Quita comentarios multilínea
    bloque_json = re.sub(r',\s*}', '}', bloque_json) # Quita trailing commas
    bloque_json = re.sub(r',\s*\]', ']', bloque_json)

    # Parser principal
    try:
        return json.loads(bloque_json)
    except json.JSONDecodeError:
        # Parser de rescate (AST) para comillas simples
        try:
            python_str = bloque_json.replace('true', 'True').replace('false', 'False').replace('null', 'None')
            datos = ast.literal_eval(python_str)
            if isinstance(datos, dict):
                return datos
        except Exception:
            pass
            
    return None

def _empty_abstract_decision() -> ScreenerDecision:
    """Short-circuit determinista para papers sin abstract (Scielo/Crossref)."""
    return ScreenerDecision(
        justification=(
            "Abstract not available; cannot evaluate eligibility from "
            "metadata alone."
        ),
        criteria_met=CriteriaMet(
            population=False, intervention=False, outcomes=False,
            study_design=False, temporal=False, language=False,
        ),
        confidence="low",
        reason="unavailable_full_text",
        decision="uncertain",
    )

async def _screen_one(
    paper: dict,
    prisma_json: str,
    model: str,
    base_url: str,
) -> ScreenerDecision | None:
    """Ejecuta una llamada LLM con validación Pydantic estructurada."""
    client = _get_client(base_url)
    
    # str.replace preserves actual JSON brackets within the prompt
    system_prompt = SCREENER_PROMPT.replace("{prisma_criteria_json}", prisma_json)
    user_msg = f"ABSTRACT:\n{paper.get('abstract', '')}"

    try:
        mensajes = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Examples:\n{SCREENER_FEWSHOT}"},
            {"role": "user",   "content": user_msg},
        ]
        
        # --- PARCHE PARA QWQ-32B ---
        if model == settings.model_32b_name:
            try:
                # 1. Le volvemos a dar las instrucciones manuales del JSON
                esquema_json = """
IMPORTANT: You are an expert data extractor. You can reason and think freely, but your FINAL output must be a valid JSON object wrapped EXACTLY inside <json> and </json> tags. Do NOT use single quotes for keys.

Example format:
<json>
{
  "justification": "Explanation of your reasoning",
  "criteria_met": {
    "population": true,
    "intervention": false,
    "outcomes": null,
    "study_design": false,
    "temporal": true,
    "language": true
  },
  "confidence": "high",
  "reason": "wrong_study_design",
  "decision": "exclude"
}
</json>
"""
                mensajes_32b = mensajes + [{"role": "system", "content": esquema_json}]
                
                raw_client = AsyncOpenAI(base_url=base_url, api_key=settings.vllm_api_key or "EMPTY")
                respuesta_cruda = await asyncio.wait_for(
                    raw_client.chat.completions.create(
                        model=model,
                        messages=mensajes_32b,
                        temperature=0.3,
                        max_tokens=4096,
                        # Quitamos extra_body porque vLLM lo está ignorando
                    ),
                    timeout=600.0,
                )
                
                texto = respuesta_cruda.choices[0].message.content
                
                if not texto:
                    raise ValueError("Texto vacío")
                    
                # 2. Usamos nuestra función limpiadora a prueba de balas
                datos = limpiar_respuesta_qwq(texto)
                
                if datos:
                    return ScreenerDecision(**datos)
                else:
                    raise ValueError("No se pudo extraer JSON válido del texto libre.")
                    
            except Exception as e:
                # Falla en silencio y deja que el 7B tome el control
                raise ValueError("Fallo en 32B")


        # --- COMPORTAMIENTO NORMAL PARA QWEN 7B ---
        else:
            return await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    response_model=ScreenerDecision,
                    messages=mensajes,
                    max_retries=3,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    temperature=0.3,
                    max_tokens=1024,
                ),
                timeout=LLM_TIMEOUT_S,
            )

    except (ValidationError, json.JSONDecodeError, asyncio.TimeoutError, ValueError):
        return None
    except Exception as e:
        logger.exception(
            f"screener LLM call failed: {str(e)}",
            extra={"paper_id": paper.get("paper_id"), "model": model, "node": "screener"},
        )
        return None

def _should_escalate(d: ScreenerDecision) -> bool:
    return d.confidence in ESCALATE_CONFIDENCES or d.decision in ESCALATE_DECISIONS

async def _cascade_screen(
    paper: dict,
    prisma_json: str,
) -> tuple[ScreenerDecision | None, str]:
    """7B → 32B en cascada."""
    if not (paper.get("abstract") or "").strip():
        return _empty_abstract_decision(), "skip_no_abstract"

    # Fase 1: Qwen 7B Fast Pass
    first = await _screen_one(
        paper, prisma_json, settings.model_7b_name, settings.vllm_url_7b,
    )
    if first is None:
        return None, "failed"

    if not _should_escalate(first):
        return first, "7b"

    # Fase 2: QwQ-32B Reasoning Fallback
    second = await _screen_one(
        paper, prisma_json, settings.model_32b_name, settings.vllm_url_32b,
    )
    if second is None:
        logger.warning(
            "screener: 32b failed, keeping 7b veredict",
            extra={"paper_id": paper.get("paper_id"), "node": "screener"},
        )
        return first, "7b_fallback"
    
    # --- MENSAJE DE ÉXITO DEL 32B ---
    logger.info(
        f"screener: 32B took the final decision succesfully for the paper {paper.get('paper_id')}",
        extra={"paper_id": paper.get("paper_id"), "node": "screener"},
    )
    # --------------------------------
    
    return second, "32b"

# --- LangGraph Node ---
async def screener_node(state: AxiomState) -> dict:
    """Nodo del grafo. Devuelve deltas para los reducers de operator.add."""
    papers = state.get("papers_found", [])
    prisma = state.get("prisma_criteria") or {}

    if not papers:
        logger.warning("screener: papers_found empty, nothing to screen", extra={"node": "screener"})
        return {}

    prisma_json = json.dumps(prisma, ensure_ascii=False)
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def _gated(p: dict):
        async with sem:
            return await _cascade_screen(p, prisma_json)

    # return_exceptions aisla fallos de papers individuales[cite: 2]
    results = await asyncio.gather(
        *(_gated(p) for p in papers),
        return_exceptions=True,
    )

    screened: list[dict] = []
    excluded: list[dict] = []
    errors:   list[dict] = []

    for paper, result in zip(papers, results):
        pid = paper.get("paper_id", "unknown_id")

        if isinstance(result, Exception):
            errors.append({
                "node": "screener",
                "paper_id": pid,
                "error": f"{type(result).__name__}: {result}",
            })
            continue

        decision, route = result
        if decision is None:
            errors.append({
                "node": "screener",
                "paper_id": pid,
                "error": "screener_failed_both_models",
            })
            continue

        screened_paper = {
            **paper,
            "screening": {**decision.model_dump(), "route": route},
        }
        
        if decision.decision == "exclude":
            excluded.append(screened_paper)
        else:
            screened.append(screened_paper)

    logger.info(
        "screener: %d incl/uncert, %d excl, %d errores (de %d papers)",
        len(screened), len(excluded), len(errors), len(papers),
    )
    
    return {
        "screened_papers": screened,
        "papers_excluded": excluded,
        "errors":          errors,
    }