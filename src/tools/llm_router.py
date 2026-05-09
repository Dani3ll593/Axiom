"""
Enrutador central de LLMs (vLLM / AMD MI300X).

Gestiona los clientes asíncronos para evitar bloquear el event loop de LangGraph
y provee utilidades para limpiar las respuestas con cadenas de razonamiento (<think>).
"""
import ast
import json
import logging
import re

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
def _extract_balanced_json(text: str) -> str | None:
    """Encuentra el primer `{` y devuelve el substring hasta la `}` que lo cierra,
    contando llaves balanceadas y respetando strings (no cuenta `{` dentro de comillas).
    Devuelve None si no encuentra un objeto bien cerrado.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _sanitize_json_block(s: str) -> str:
    """Limpia ruido común en outputs de LLM antes de json.loads."""
    s = s.replace("\xa0", " ").replace("\t", " ")
    s = re.sub(r"//.*", "", s)                        # // comentarios línea
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)  # /* */ comentarios bloque
    s = re.sub(r",\s*([}\]])", r"\1", s)              # trailing commas
    return s.strip()


def extract_json_from_response(raw: str) -> dict:
    """Extrae JSON de la respuesta del LLM, robusto a múltiples formatos:

    - `<json>...</json>` (preferido si el prompt lo pide)
    - ```` ```json ... ``` ```` o ```` ``` ... ``` ```` (markdown fence)
    - `{...}` en texto libre (búsqueda de llaves balanceadas)

    Los bloques `<think>...</think>` de QwQ-32B se descartan antes de buscar.
    Si encuentra un candidato pero `json.loads` falla, intenta `ast.literal_eval`
    como rescate (cubre comillas simples y otros casos cuasi-Python).

    NOTE: el screener tiene una función equivalente `limpiar_respuesta_qwq` con
    heurísticas un poco distintas. A futuro convendría unificarlas.

    Raises:
        ValueError: si raw está vacío.
        json.JSONDecodeError: si no se logra extraer JSON válido. El mensaje
            incluye un preview de los primeros 500 chars de la respuesta cruda
            para diagnosticar qué devolvió el modelo.
    """
    if not raw or not raw.strip():
        raise ValueError("Received empty response from LLM")

    # 1. Quitar bloques <think>...</think> (incluso múltiples). Si hay <think>
    #    sin cerrar, sobrevive en el texto y caerá en error con preview claro.
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)

    # 2. Reunir candidatos en orden de preferencia.
    candidates: list[str] = []

    m = re.search(r"<json>\s*(\{.*?\})\s*</json>", text, re.DOTALL | re.IGNORECASE)
    if m:
        candidates.append(m.group(1))

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        candidates.append(m.group(1))

    balanced = _extract_balanced_json(text)
    if balanced:
        candidates.append(balanced)

    # 3. Probar cada candidato con sanitización + parser principal + rescate AST.
    for c in candidates:
        cleaned = _sanitize_json_block(c)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                py_compat = (
                    cleaned.replace("true", "True")
                           .replace("false", "False")
                           .replace("null", "None")
                )
                parsed = ast.literal_eval(py_compat)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue

    # 4. Todo falló: error con preview para diagnosticar.
    preview = raw[:500].replace("\n", " ")
    logger.error(
        f"extract_json_from_response: no parseable JSON encontrado. "
        f"Raw preview (500 chars): {preview!r}"
    )
    raise json.JSONDecodeError(
        f"No se pudo extraer JSON válido. Raw preview: {preview!r}",
        raw, 0,
    )


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