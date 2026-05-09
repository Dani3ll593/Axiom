"""
Configuración global de Axiom.
Carga las variables desde el archivo .env de forma tipada y segura.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # ─── APIs de Búsqueda y Extracción ───
    pubmed_api_key: str | None = None
    openalex_api_key: str | None = None
    # El email es obligatorio para los "polite pools". Falla si está vacío.
    contact_email: str = Field(..., min_length=1)

    # ─── Servidores Locales (vLLM en AMD MI300X) ───
    vllm_url_7b: str = "VLLM_URL_7B"
    vllm_url_32b: str = "VLLM_URL_32B"
    vllm_api_key: str | None = None

    model_7b_name: str = "Qwen/Qwen2.5-7B-Instruct"
    model_32b_name: str = "Qwen/QwQ-32B-Preview"

    # ─── Rutas del Sistema ───
    chroma_persist_dir: str = "./data/chroma_db"

 # ─── Clusterer (BGE-M3 + AgglomerativeClustering) ───
    # Métrica coseno: threshold=distancia, NO similitud.
    #   0.30-0.40 → near-duplicates (muy estricto, muchos singletons)
    #   0.50      → mismo subtopic (default sensato para BGE-M3)
    #   0.60-0.70 → mismo dominio general (laxo, clusters grandes)
    cluster_distance_threshold: float = 0.7

    # Cota dura sobre el JSON serializado del cluster que se manda al analyst.
    # El 7B es el más estricto: ctx=8192 − max_tokens(2048) − system_prompt(~1400)
    # − margen(~300) ≈ 4500 tokens ≈ 16K chars. Si el JSON pruned excede esto,
    # el clusterer parte el cluster en sub-clusters consecutivos. Evita overflow
    # de context window aunque el threshold semántico produzca clusters densos.
    analyst_max_user_chars: int = 16000

    # ─── UI y Streamlit ───
    streamlit_server_port: int = 8501

    # Permite ignorar variables extra en el .env que no usemos aquí
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

# Instancia global (singleton) para importar desde otros módulos
settings = Settings()