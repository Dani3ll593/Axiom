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