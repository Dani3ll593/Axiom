"""
Enrutador central de LLMs (vLLM / AMD MI300X).

Gestiona los clientes asíncronos para evitar bloquear el event loop de LangGraph
y provee utilidades para limpiar las respuestas con cadenas de razonamiento (<think>).
"""
import re
import json
import logging
from openai import AsyncOpenAI

from src.config import settings

logger = logging.getLogger(__name__)

# --- 1. Inicialización de Clientes (Lazy / Global) ---
# vLLM requiere la key si está configurada, o "EMPTY" si no hay auth[cite: 6]
_api_key = settings.vllm_api_key or "EMPTY"

LLM_7B = AsyncOpenAI(
    base_url=settings.vllm_url_7b,
    api_key=_api_key,
    timeout=120.0,
)

LLM_32B = AsyncOpenAI(
    base_url=settings.vllm_url_32b,
    api_key=_api_key,
    timeout=300.0,
)

# --- 2. Utilidades de Parseo ---
def extract_json_from_response(raw: str) -> dict:
    """
    Extrae JSON de las respuestas del LLM, ignorando los bloques <think>
    que genera QwQ-32B de forma nativa[cite: 6].
    """
    if not raw:
        raise ValueError("Received empty response from LLM")

    # Descarta el bloque <think>...</think> y captura el contenido real[cite: 6]
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    
    # Por si el modelo añade las clásicas vallas de markdown ```json ... ```
    cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned).rstrip("`").strip()
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Error decodificando JSON tras limpiar <think>: {e}")
        logger.debug(f"Raw string: {raw}")
        raise


# --- 3. Enrutador Principal ---
async def route_task(task_type: str, messages: list[dict], **kwargs) -> str:
    """
    Enruta cada tarea al modelo adecuado según la complejidad cognitiva requerida[cite: 7].
    
    Args:
        task_type: Categoría de la tarea (ej. 'abstract_screening', 'gap_identification').
        messages: Lista de mensajes formato OpenAI.
        **kwargs: Parámetros adicionales (temperature, max_tokens, etc.)
        
    Returns:
        str: El contenido crudo de la respuesta del LLM.
    """
    # Definimos el ruteo estricto basado en el documento de arquitectura[cite: 7]
    routing = {
        "search_decomposition":    settings.model_7b_name,
        "abstract_screening":      settings.model_7b_name,
        "pdf_extraction":          settings.model_7b_name,
        "contradiction_detection": settings.model_32b_name,
        "gap_identification":      settings.model_32b_name,
        "narrative_generation":    settings.model_32b_name,
    }

    # Por defecto, si no conocemos la tarea, usamos el modelo grande por seguridad
    model_name = routing.get(task_type, settings.model_32b_name)
    
    # Asignamos el cliente correcto
    client = LLM_7B if model_name == settings.model_7b_name else LLM_32B

    logger.info(f"route_task: Enrutando '{task_type}' al modelo {model_name}")

    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        **kwargs
    )
    
    return response.choices[0].message.content