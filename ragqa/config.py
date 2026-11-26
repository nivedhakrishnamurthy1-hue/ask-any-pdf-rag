"""Central configuration. Every value can be overridden from the .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

# MiniLM truncates input at ~256 tokens (about 180-190 words), so chunks stay below that.
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "180"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))

TOP_K = int(os.getenv("TOP_K", "4"))
# If even the best chunk is less similar than this, we refuse without calling the LLM.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.20"))

# Limits that protect a public deployment (see app.py)
MAX_PAGES = int(os.getenv("MAX_PAGES", "150"))
MAX_QUESTIONS_PER_SESSION = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "25"))
