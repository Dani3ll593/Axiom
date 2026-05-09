"""
Módulo de parseo de PDFs a texto crudo.
Implementación primaria con PyMuPDF (fitz). Degrada gracefully a 'abstract_only'
si el PDF es una imagen escaneada o está corrupto.
"""
import logging
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Umbral de caracteres. Si un PDF produce menos que esto, asumimos que es 
# un documento escaneado (imágenes) y no texto extraíble.
MIN_TEXT_LENGTH = 1200

def parse_pdf(pdf_bytes: bytes) -> dict:
    """
    Extrae texto crudo de un array de bytes de un PDF.
    
    Args:
        pdf_bytes (bytes): El contenido crudo del archivo PDF.
        
    Returns:
        dict: {
            "strategy": "pymupdf_full" | "abstract_only",
            "text": str | None,
            "error": str | None
        }
    """
    if not pdf_bytes:
        return {"strategy": "abstract_only", "text": None, "error": "empty_bytes_received"}

    try:
        # Abrimos el stream en memoria
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Extracción estándar de texto
        pages_text = [page.get_text() for page in doc]
        full_text = "\n".join(pages_text)
        
        # Si el texto es demasiado corto, es muy probable que sea un escaneo sin OCR[cite: 7]
        if len(full_text.strip()) < MIN_TEXT_LENGTH:
            logger.warning(
                "PDF parsing yielded insufficient text (likely scanned without OCR). "
                "Falling back to abstract."
            )
            return {"strategy": "abstract_only", "text": None, "error": "insufficient_text_length"}
            
        return {"strategy": "pymupdf_full", "text": full_text, "error": None}

    except Exception as e:
        logger.error(f"Failed to parse PDF: {type(e).__name__} - {str(e)}")
        # Nunca rompemos la ejecución. El fallback del Extractor usará el abstract[cite: 7].
        return {"strategy": "abstract_only", "text": None, "error": str(e)}