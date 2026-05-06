"""
Módulo de verificación de acceso abierto (Open Access).
Regla de negocio: Un paper se considera abierto si 2 de 3 fuentes lo confirman.
"""
import asyncio
import logging
import httpx

from src.config import settings

logger = logging.getLogger(__name__)

async def check_access_async(doi: str) -> dict:
    """
    Consulta Unpaywall, OpenAlex y Crossref concurrentemente.
    Retorna un diccionario con el estado de acceso y la URL del PDF (si existe).
    """
    email = settings.contact_email
    openalex_key = settings.openalex_api_key
    
    results = {
        "unpaywall": False,
        "openalex": False,
        "crossref": False,
        "pdf_url": None
    }

    if not doi:
        return {"is_open": False, "pdf_url": None, "confidence": 0.0}

    # Timeout estricto de 10s para evitar bloquear el pipeline
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Crossref y Unpaywall exigen "polite pools" usando un email institucional
        headers = {"User-Agent": f"Axiom/1.0 (mailto:{email})"}
        
        # Parámetros para OpenAlex (la API key ahora va como query param, no header)[cite: 7]
        oa_params = {"api_key": openalex_key} if openalex_key else {}

        # Disparar las 3 peticiones en paralelo
        req_unpaywall = client.get(f"https://api.unpaywall.org/v2/{doi}?email={email}")
        req_openalex = client.get(f"https://api.openalex.org/works/https://doi.org/{doi}", params=oa_params)
        req_crossref = client.get(f"https://api.crossref.org/works/{doi}", headers=headers)

        unpaywall_r, openalex_r, crossref_r = await asyncio.gather(
            req_unpaywall, req_openalex, req_crossref, return_exceptions=True
        )

    # 1. Parsear Unpaywall
    if isinstance(unpaywall_r, httpx.Response) and unpaywall_r.is_success:
        data = unpaywall_r.json()
        results["unpaywall"] = data.get("is_oa", False)
        # Extraer el mejor link disponible para el PDF
        best_oa = data.get("best_oa_location") or {}
        results["pdf_url"] = best_oa.get("url_for_pdf")

    # 2. Parsear OpenAlex
    if isinstance(openalex_r, httpx.Response) and openalex_r.is_success:
        data = openalex_r.json()
        results["openalex"] = data.get("open_access", {}).get("is_oa", False)
        # Si Unpaywall no nos dio PDF, intentamos sacar el de OpenAlex
        if not results["pdf_url"]:
            results["pdf_url"] = data.get("open_access", {}).get("oa_url")

    # 3. Parsear Crossref
    if isinstance(crossref_r, httpx.Response) and crossref_r.is_success:
        data = crossref_r.json().get("message", {})
        # Crossref indica OA mediante la presencia de licencias específicas
        licenses = data.get("license", [])
        results["crossref"] = len(licenses) > 0

    # Lógica de Consenso: Regla 2 de 3[cite: 7]
    votes_for_open = sum([
        results["unpaywall"], 
        results["openalex"], 
        results["crossref"]
    ])
    
    is_open = votes_for_open >= 2

    # Si decidimos que es cerrado, limpiamos la URL por seguridad
    if not is_open:
        results["pdf_url"] = None

    return {
        "is_open": is_open,
        "pdf_url": results["pdf_url"],
        "confidence": votes_for_open / 3.0,
        "votes": {
            "unpaywall": results["unpaywall"],
            "openalex": results["openalex"],
            "crossref": results["crossref"]
        }
    }