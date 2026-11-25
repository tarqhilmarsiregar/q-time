
import argparse
import hashlib
import logging
import re

from datetime import date
from pathlib import Path

import pymupdf

from pgvector import Vector
from psycopg.types.json import Jsonb

from app.rag.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
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
    overlap: int = CHUNK_OVERLAP
):
    """
    Split text into overlapping character chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size."
        )

    # Normalize unnecessary whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    chunks = []
    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        # Try to split at a whitespace boundary
        if end < len(text):

            boundary = text.rfind(
                " ",
                start + int(chunk_size * 0.7),
                end
            )

            if boundary > start:
                end = boundary

        content = text[start:end].strip()

        if content:
            chunks.append(content)

        if end >= len(text):
            break

        # Maintain a small overlap between chunks
        start = max(
            end - overlap,
            start + 1
        )

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

    # Generate vector representations
    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    vectors = embed_passages(texts)

    logger.info(
        "Embedding generation completed."
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
