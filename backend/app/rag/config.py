
import os
from pathlib import Path

from dotenv import load_dotenv


# Resolve project directories
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent

# Load environment variables from backend/.env
load_dotenv(BACKEND_DIR / ".env")


# Database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://academic_admin:local_dev_password"
    "@localhost:5432/academic_copilot"
)


# Embedding configuration
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-small"
)

EMBEDDING_DIMENSION = int(
    os.getenv("EMBEDDING_DIMENSION", "384")
)

EMBEDDING_BATCH_SIZE = int(
    os.getenv("EMBEDDING_BATCH_SIZE", "16")
)


# Chunking configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

TOP_K = int(os.getenv("TOP_K", "5"))


# Document directory
DOCUMENTS_DIR = PROJECT_DIR / "data" / "raw" / "academic_documents"

DOCUMENTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEFAULT_ACADEMIC_YEAR = os.getenv(
    "DEFAULT_ACADEMIC_YEAR",
    "2026/2027"
)