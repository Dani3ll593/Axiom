"""
Prompt loader for Axiom agents.

All prompts live as `.txt` or `.json` files alongside this module so they can
be edited without touching Python code. The loader reads them once at import
time and caches the strings in module-level constants. JSON files are loaded
into Python dicts.

Usage from agent modules:

    from src.prompts import (
        SEARCHER_PROMPT,
        SCREENER_PROMPT,
        SCREENER_FEWSHOT,
        EXTRACTION_PROMPT,
        EXTRACTOR_SCHEMA,
        PRISMA_CRITERIA_TEMPLATE,
    )

The loader fails loudly at import time if any expected file is missing.
This is deliberate: a missing prompt should never reach runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def _read_text(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}. "
            f"Every prompt declared in src/prompts/__init__.py must exist on disk."
        )
    return path.read_text(encoding="utf-8").strip()


def _read_json(filename: str) -> dict:
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"JSON prompt asset not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# Agente 1 — Searcher
SEARCHER_PROMPT: str = _read_text("searcher_prompt.txt")

# Agente 2 — Screener
SCREENER_PROMPT:  str = _read_text("screener_prompt.txt")
SCREENER_FEWSHOT: str = _read_text("screener_fewshot.txt")

# Agente 3 — Extractor
# EXTRACTION_PROMPT is a template with a {schema} placeholder. The agent
# renders it at call time with PaperExtraction.model_json_schema().
# EXTRACTOR_SCHEMA is the JSON reference copy (NOT the source of truth —
# the source of truth is the Pydantic class in src/agents/extractor.py).
EXTRACTION_PROMPT: str  = _read_text("extraction_prompt.txt")
EXTRACTOR_SCHEMA:  dict = _read_json("extractor_schema.json")

# Agentes 4a / 4b — Analysts (dual)
ANALYST_PROMPT_7B:  str = _read_text("analyst_prompt_7b.txt")
ANALYST_PROMPT_32B: str = _read_text("analyst_prompt_32b.txt")

# Agente 5 — Gap Finder
GAPFINDER_PROMPT: str = _read_text("gapfinder_prompt.txt")

# Agente 6 — Writer
WRITER_PROMPT:     str = _read_text("writer_prompt.txt")
WRITER_APA7_RULES: str = _read_text("writer_apa7_rules.txt")

# UI helper — default PRISMA criteria offered to the user in Streamlit
PRISMA_CRITERIA_TEMPLATE: dict = _read_json("prisma_criteria_template.json")


__all__ = [
    "SEARCHER_PROMPT",
    "SCREENER_PROMPT",
    "SCREENER_FEWSHOT",
    "EXTRACTION_PROMPT",
    "EXTRACTOR_SCHEMA",
    "ANALYST_PROMPT_7B",
    "ANALYST_PROMPT_32B",
    "GAPFINDER_PROMPT",
    "WRITER_PROMPT",
    "WRITER_APA7_RULES",
    "PRISMA_CRITERIA_TEMPLATE",
]