import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

XAI_API_KEY = os.getenv("XAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "xai")
LLM_MODEL = os.getenv("LLM_MODEL", "grok-4-0709")

PROVIDER_DEFAULTS = {
    "xai": {"model": "grok-4-0709", "key_env": "XAI_API_KEY"},
    "google": {"model": "gemini-2.5-flash", "key_env": "GOOGLE_API_KEY"},
    "anthropic": {"model": "claude-sonnet-4-20250514", "key_env": "ANTHROPIC_API_KEY"},
}

RATE_LIMIT_DELAY = 0.12 if NCBI_API_KEY else 0.35

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ELINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"

PMC_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

DB_PATH = Path(__file__).parent / "pubmed_research.db"
OUTPUT_DIR = Path(__file__).parent / "output"

MAX_WORKERS = 8
EFETCH_BATCH_SIZE = 200
BULK_INSERT_BATCH_SIZE = 500


def get_api_key(provider: str) -> str:
    keys = {
        "xai": XAI_API_KEY,
        "google": GOOGLE_API_KEY,
        "anthropic": ANTHROPIC_API_KEY,
    }
    return keys.get(provider, "")


def get_default_model(provider: str) -> str:
    return PROVIDER_DEFAULTS.get(provider, {}).get("model", "")
