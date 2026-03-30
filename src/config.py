# ABOUTME: Central configuration for paths, model names, and API settings.
# ABOUTME: All project-wide constants live here to avoid magic strings.

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
INVENTORY_PATH = DATA_DIR / "inventory.csv"
PROCESSED_TEXTS_PATH = DATA_DIR / "processed_texts.json"
CHUNKS_PATH = DATA_DIR / "chunks.json"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
UNANSWERED_LOG_PATH = PROJECT_ROOT / "unanswered_log.json"

# Source corpus — reference in place, never copy
CORPUS_DIR = Path(
    "/Users/eduardolizana/Documents/6. Companies "
    "/1. Companies with Rodrigo/KeepMeCompany/"
)

# File types to skip during inventory
SKIP_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".wav",
    ".mp3",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
}

# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "qwen/qwen3.5-flash-02-23"

# Retrieval defaults (tuned in Phase 4)
DEFAULT_VECTOR_WEIGHT = 0.6
DEFAULT_BM25_WEIGHT = 0.4
DEFAULT_K = 5
DEFAULT_FETCH_K = 20  # Larger candidate set for MMR diversity filtering
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 20  # Fetch this many from hybrid, rerank to top k

# Chunking defaults (tuned in Phase 4)
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
