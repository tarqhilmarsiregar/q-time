
import argparse
import hashlib
import logging
import re

from datetime import date
from pathlib import Path

import pymupdf

import numpy as np

from pgvector import Vector
from psycopg.types.json import Jsonb

from app.rag.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_DIMENSION,
)

from app.rag.database import get_connection
from app.rag.embedding import embed_passages


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================
# 1. PDF TEXT EXTRACTION
# ==========================================

def extract_pdf_pages(pdf_path: Path):
    """
    Extract text from each PDF page.
    Page number starts from 1.
    """

    pages = []

    with pymupdf.open(pdf_path) as document:

        if document.is_encrypted:
            raise ValueError(
                "PDF is encrypted and requires a password."
            )

        total_pages = len(document)

        for page_index, page in enumerate(document):

            text = page.get_text(
                "text",
                sort=True
            ).strip()

            if text:
                pages.append({
                    "page_number": page_index + 1,
                    "text": text
                })

    if not pages:
        raise ValueError(
            "No extractable text found. "
            "The PDF may be scanned or image-based."
        )

    logger.info(
        "Extracted text from %s of %s pages.",
        len(pages),
        total_pages
    )

    return pages, total_pages


# ==========================================
# 2. TEXT CHUNKING
# ==========================================

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks without splitting words.

    Chunk size and overlap are measured in characters.
    """

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be greater than or equal to zero "
            "and smaller than chunk_size."
        )

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:

        target_end = min(start + chunk_size, text_length)

        # Find a whitespace boundary near the target end.
        if target_end < text_length:
            min_boundary = start + int(chunk_size * 0.7)

            boundary = text.rfind(
                " ",
                min_boundary,
                target_end,
            )

            end = boundary if boundary > start else target_end

        else:
            end = target_end

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # Calculate the next starting position using overlap.
        next_start = max(end - overlap, start + 1)

        # Move forward until the next complete word boundary.
        while next_start < text_length and text[next_start] != " ":
            next_start += 1

        if next_start >= text_length:
            break

        start = next_start + 1

    return chunks


# ==========================================
# 3. DOCUMENT CHUNKING
# ==========================================

def create_document_chunks(pages):

    document_chunks = []

    chunk_index = 0

    for page in pages:

        page_chunks = chunk_text(
            page["text"]
        )

        for content in page_chunks:

            document_chunks.append({
                "chunk_index": chunk_index,
                "page_number": page["page_number"],
                "content": content
            })

            chunk_index += 1

    return document_chunks


# ==========================================
# 4. FILE HASHING
# ==========================================

def calculate_sha256(file_path: Path):

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:

        for block in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            sha256.update(block)

    return sha256.hexdigest()


# ==========================================
# DOCUMENT METADATA VALIDATION
# ==========================================

def validate_document_metadata(args, file_path):

    title = file_path.stem.replace(
        "_", " "
    ).title()

    file_name = file_path.name
    category = args.category
    academic_year = args.academic_year
    version = args.version
    status = args.status

    if not title.strip():
        raise ValueError("Document title cannot be empty.")

    if len(title) > 255:
        raise ValueError(
            "Document title exceeds 255 characters."
        )

    if len(file_name) > 255:
        raise ValueError(
            "File name exceeds 255 characters."
        )

    if not category or not category.strip():
        raise ValueError(
            "Document category cannot be empty."
        )

    if len(category) > 100:
        raise ValueError(
            "Document category exceeds 100 characters."
        )

    if academic_year and len(academic_year) > 20:
        raise ValueError(
            "Academic year exceeds 20 characters."
        )

    if version and len(version) > 50:
        raise ValueError(
            "Document version exceeds 50 characters."
        )

    if status not in ["draft", "published"]:
        raise ValueError(
            "Invalid document status."
        )

    logger.info(
        "Document metadata validation successful."
    )

def validate_document_metadata(args, file_path: Path) -> date | None:
    """
    Validate and normalize document metadata
    before ingestion and database persistence.

    Validation follows the documents table schema.
    """

    errors = []

    # Generate metadata derived from the PDF file.
    title = file_path.stem.replace("_", " ").title()
    file_name = file_path.name

    # Normalize optional and required metadata.
    args.category = (args.category or "").strip()
    args.academic_year = (
        args.academic_year.strip()
        if args.academic_year and args.academic_year.strip()
        else None
    )
    args.version = (
        args.version.strip()
        if args.version and args.version.strip()
        else None
    )
    args.status = (args.status or "").strip().lower()

    # Validate required text fields.
    if not title:
        errors.append("Document title cannot be empty.")

    if not file_name:
        errors.append("File name cannot be empty.")

    if not args.category:
        errors.append("Document category cannot be empty.")

    # Validate VARCHAR length constraints.
    if len(title) > 255:
        errors.append(
            f"Title exceeds 255 characters: {len(title)}."
        )

    if len(file_name) > 255:
        errors.append(
            f"File name exceeds 255 characters: {len(file_name)}."
        )

    if len(args.category) > 100:
        errors.append(
            f"Category exceeds 100 characters: {len(args.category)}."
        )

    if args.academic_year and len(args.academic_year) > 20:
        errors.append(
            "Academic year exceeds 20 characters."
        )

    if args.version and len(args.version) > 50:
        errors.append(
            "Version exceeds 50 characters."
        )

    # Validate document status.
    allowed_statuses = {"draft", "published"}

    if args.status not in allowed_statuses:
        errors.append(
            f"Invalid document status: {args.status}. "
            f"Allowed values: {', '.join(sorted(allowed_statuses))}."
        )

    # Validate effective date.
    effective_date = None

    if args.effective_date:
        try:
            effective_date = date.fromisoformat(
                args.effective_date
            )
        except ValueError:
            errors.append(
                "Invalid effective date. "
                "Expected format: YYYY-MM-DD."
            )

    # Raise all validation errors together.
    if errors:
        error_message = "\n".join(
            f"- {error}" for error in errors
        )

        raise ValueError(
            f"Document metadata validation failed:\n{error_message}"
        )

    logger.info("Document metadata validation passed.")

    return effective_date

def validate_embeddings(
    embeddings,
    expected_count: int,
    expected_dimension: int = EMBEDDING_DIMENSION,
) -> np.ndarray:
    """
    Validate generated embedding vectors before persistence.

    Checks:
    1. Embedding count matches the number of chunks.
    2. Embedding dimension matches the configured dimension.
    3. All values are finite.
    4. No embedding vector is a zero vector.
    """

    if expected_count <= 0:
        raise ValueError(
            "Expected embedding count must be greater than zero."
        )

    if expected_dimension <= 0:
        raise ValueError(
            "Expected embedding dimension must be greater than zero."
        )

    try:
        vectors = np.asarray(
            embeddings,
            dtype=np.float32,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Embeddings cannot be converted into a numeric array."
        ) from exc

    # Check array structure.
    if vectors.ndim != 2:
        raise ValueError(
            "Invalid embedding structure. "
            f"Expected a 2D array, received {vectors.ndim}D."
        )

    actual_count, actual_dimension = vectors.shape

    # Check embedding count.
    if actual_count != expected_count:
        raise ValueError(
            "Embedding count mismatch. "
            f"Expected {expected_count}, received {actual_count}."
        )

    # Check embedding dimension.
    if actual_dimension != expected_dimension:
        raise ValueError(
            "Embedding dimension mismatch. "
            f"Expected {expected_dimension}, "
            f"received {actual_dimension}."
        )

    # Check NaN and Infinity.
    if not np.isfinite(vectors).all():
        raise ValueError(
            "Embeddings contain NaN or Infinity values."
        )

    # Check zero vectors.
    vector_norms = np.linalg.norm(
        vectors,
        axis=1,
    )

    if np.any(vector_norms == 0):
        raise ValueError(
            "Embeddings contain zero vectors."
        )

    logger.info(
        "Embedding validation passed. "
        "Count: %s, Dimension: %s",
        actual_count,
        actual_dimension,
    )

    return vectors

# ==========================================
# 5. DATABASE STORAGE
# ==========================================

def save_document(
    file_path: Path,
    pages,
    total_pages: int,
    chunks,
    vectors,
    category: str,
    academic_year: str,
    version: str,
    status: str,
    effective_date
):
    content_hash = calculate_sha256(
        file_path
    )

    title = file_path.stem.replace(
        "_", " "
    ).title()

    with get_connection() as conn:

        # Prevent duplicate document ingestion
        existing = conn.execute(
            """
            SELECT id
            FROM documents
            WHERE content_sha256 = %s
            """,
            (content_hash,)
        ).fetchone()

        if existing:

            logger.warning(
                "Document already exists. ID: %s",
                existing[0]
            )

            return None

        document_row = conn.execute(
            """
            INSERT INTO documents (
                title,
                file_name,
                category,
                academic_year,
                version,
                effective_date,
                status,
                content_sha256,
                page_count
            )
            VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s
            )
            RETURNING id
            """,
            (
                title,
                file_path.name,
                category,
                academic_year,
                version,
                effective_date,
                status,
                content_hash,
                total_pages
            )
        ).fetchone()

        document_id = document_row[0]

        rows = []

        for chunk, vector in zip(chunks, vectors):

            rows.append((
                document_id,
                chunk["chunk_index"],
                chunk["page_number"],
                chunk["content"],
                Vector(vector.tolist()),
                Jsonb({
                    "source": "pdf",
                    "file_name": file_path.name,
                    "extraction": "pymupdf"
                })
            ))


        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO document_chunks (
                    document_id,
                    chunk_index,
                    page_number,
                    content,
                    embedding,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                rows
            )
        
    return document_id


# ==========================================
# 6. MAIN INGESTION PROCESS
# ==========================================

def ingest_document(args):

    file_path = Path(args.file).resolve()

    if not file_path.exists():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(
            "Only PDF documents are supported."
        )

    # Validate document metadata before processing.
    effective_date = validate_document_metadata(
        args,
        file_path,
    )

    logger.info(
        "Starting ingestion: %s",
        file_path.name
    )

    pages, total_pages = extract_pdf_pages(
        file_path
    )

    chunks = create_document_chunks(pages)

    if not chunks:
        raise ValueError(
            "No document chunks were generated."
        )

    logger.info(
        "Generated %s chunks.",
        len(chunks)
    )

        # Generate vector representations.
    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    vectors = embed_passages(texts)

    # Validate embeddings before database persistence.
    vectors = validate_embeddings(
        embeddings=vectors,
        expected_count=len(chunks),
    )

    logger.info(
        "Embedding generation completed: %s vectors, dimension %s.",
        len(vectors),
        EMBEDDING_DIMENSION,
    )

    effective_date = (
        date.fromisoformat(args.effective_date)
        if args.effective_date
        else None
    )

    document_id = save_document(
        file_path=file_path,
        pages=pages,
        total_pages=total_pages,
        chunks=chunks,
        vectors=vectors,
        category=args.category,
        academic_year=args.academic_year,
        version=args.version,
        status=args.status,
        effective_date=effective_date
    )

    if document_id is None:
        return

    logger.info(
        "Document ingestion successful."
    )

    logger.info(
        "Document ID: %s",
        document_id
    )

    logger.info(
        "Total chunks stored: %s",
        len(chunks)
    )


# ==========================================
# 7. CLI
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="Academic Document Ingestion"
    )

    parser.add_argument(
        "--file",
        required=True,
        help="Path to academic PDF"
    )

    parser.add_argument(
        "--category",
        default="academic-general"
    )

    parser.add_argument(
        "--academic-year",
        default=None
    )

    parser.add_argument(
        "--version",
        default="1.0"
    )

    parser.add_argument(
        "--status",
        choices=["draft", "published"],
        default="draft"
    )

    parser.add_argument(
        "--effective-date",
        default=None,
        help="Format: YYYY-MM-DD"
    )

    args = parser.parse_args()

    try:

        ingest_document(args)

    except Exception:

        logger.exception(
            "Document ingestion failed."
        )

        raise


if __name__ == "__main__":
    main()
