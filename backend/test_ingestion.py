
from pathlib import Path

from app.rag.ingestion import (
    extract_pdf_pages,
    create_document_chunks,
)

from argparse import Namespace
from datetime import date
from app.rag.ingestion import validate_document_metadata

import numpy as np

from app.rag.config import EMBEDDING_DIMENSION
from app.rag.embedding import embed_passages

from app.rag.ingestion import validate_embeddings


PDF_PATH = Path(
    "../data/raw/academic_documents/"
    "KALENDER_AKADEMIK_TAHUN AKADEMIK_2026-2027_STMIK_TIME.pdf"
).resolve()

from app.rag.ingestion import chunk_text

def test_real_embedding_generation(chunks):

    print("\n[7] REAL EMBEDDING GENERATION TEST")

    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    embeddings = embed_passages(texts)

    validated_embeddings = validate_embeddings(
        embeddings=embeddings,
        expected_count=len(chunks),
    )

    print("Embedding model  :", "intfloat/multilingual-e5-small")
    print("Input chunks     :", len(chunks))
    print("Embedding count  :", len(validated_embeddings))
    print("Embedding dimension:", validated_embeddings.shape[1])

    print("Real embedding generation test: PASSED")

def test_embedding_validation():

    print("\n[6] EMBEDDING VALIDATION TEST")

    # Test 1: Valid embeddings.
    valid_embeddings = np.ones(
        (2, EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    result = validate_embeddings(
        valid_embeddings,
        expected_count=2,
    )

    assert result.shape == (
        2,
        EMBEDDING_DIMENSION,
    )

    print("[PASS] Valid embeddings accepted.")

    # Test 2: Invalid embedding count.
    try:
        validate_embeddings(
            valid_embeddings,
            expected_count=3,
        )

        raise AssertionError(
            "Invalid embedding count was not rejected."
        )

    except ValueError as error:
        assert "count mismatch" in str(error).lower()
        print("[PASS] Invalid embedding count rejected.")

    # Test 3: Invalid embedding dimension.
    invalid_dimension = np.ones(
        (2, EMBEDDING_DIMENSION - 1),
        dtype=np.float32,
    )

    try:
        validate_embeddings(
            invalid_dimension,
            expected_count=2,
        )

        raise AssertionError(
            "Invalid embedding dimension was not rejected."
        )

    except ValueError as error:
        assert "dimension mismatch" in str(error).lower()
        print("[PASS] Invalid embedding dimension rejected.")

    # Test 4: NaN values.
    invalid_nan = np.ones(
        (2, EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    invalid_nan[0, 0] = np.nan

    try:
        validate_embeddings(
            invalid_nan,
            expected_count=2,
        )

        raise AssertionError(
            "NaN embedding was not rejected."
        )

    except ValueError as error:
        assert "NaN or Infinity" in str(error)
        print("[PASS] NaN values rejected.")

    # Test 5: Zero vector.
    invalid_zero = np.ones(
        (2, EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    invalid_zero[0] = 0

    try:
        validate_embeddings(
            invalid_zero,
            expected_count=2,
        )

        raise AssertionError(
            "Zero embedding was not rejected."
        )

    except ValueError as error:
        assert "zero vectors" in str(error)
        print("[PASS] Zero vectors rejected.")

    print("Embedding validation test: PASSED")

def test_metadata_validation():

    print("\n[5] METADATA VALIDATION TEST")

    # Test 1: Valid metadata.
    valid_args = Namespace(
        category="academic-general",
        academic_year="2026/2027",
        version="1.0",
        status="draft",
        effective_date="2026-06-24",
    )

    effective_date = validate_document_metadata(
        valid_args,
        Path("pedoman_akademik.pdf"),
    )

    assert effective_date == date(2026, 6, 24)

    print("[PASS] Valid metadata accepted.")

    # Test 2: Invalid category length.
    invalid_args = Namespace(
        category="A" * 101,
        academic_year="2026/2027",
        version="1.0",
        status="draft",
        effective_date="2026-06-24",
    )

    try:
        validate_document_metadata(
            invalid_args,
            Path("pedoman_akademik.pdf"),
        )

        raise AssertionError(
            "Invalid category length was not rejected."
        )

    except ValueError as error:
        assert "Category exceeds 100 characters" in str(error)
        print("[PASS] Invalid category length rejected.")

    # Test 3: Invalid effective date.
    invalid_date_args = Namespace(
        category="academic-general",
        academic_year="2026/2027",
        version="1.0",
        status="draft",
        effective_date="2026-99-99",
    )

    try:
        validate_document_metadata(
            invalid_date_args,
            Path("pedoman_akademik.pdf"),
        )

        raise AssertionError(
            "Invalid effective date was not rejected."
        )

    except ValueError as error:
        assert "Invalid effective date" in str(error)
        print("[PASS] Invalid effective date rejected.")

    print("Metadata validation test: PASSED")


def test_chunk_integrity():
    """
    Test that chunking preserves all tokens
    and does not exceed the configured chunk size.
    """

    tokens = [f"token_{i}" for i in range(500)]
    original_text = " ".join(tokens)

    chunks = chunk_text(
        original_text,
        chunk_size=1000,
        overlap=150,
    )

    extracted_tokens = set()

    for chunk in chunks:
        extracted_tokens.update(chunk.split())

    expected_tokens = set(tokens)
    missing_tokens = expected_tokens - extracted_tokens

    oversized_chunks = [
        len(chunk)
        for chunk in chunks
        if len(chunk) > 1000
    ]

    print("\n[4] CHUNK INTEGRITY TEST")
    print("Original tokens :", len(expected_tokens))
    print("Generated chunks:", len(chunks))
    print("Missing tokens  :", len(missing_tokens))
    print("Oversized chunks:", len(oversized_chunks))

    if missing_tokens:
        print("Missing examples:", list(missing_tokens)[:10])

    assert not missing_tokens, (
        f"Chunking lost {len(missing_tokens)} tokens."
    )

    assert not oversized_chunks, (
        "Some chunks exceed the configured chunk size."
    )

    print("Chunk integrity test: PASSED")


def main():

    print("=" * 60)
    print("INGESTION PIPELINE TEST")
    print("=" * 60)

    if not PDF_PATH.exists():
        raise FileNotFoundError(PDF_PATH)

    # 1. Test PDF extraction
    pages, total_pages = extract_pdf_pages(PDF_PATH)

    print("\n[1] PDF EXTRACTION")
    print("Total PDF pages :", total_pages)
    print("Extracted pages :", len(pages))

    for page in pages:
        print(
            f"Page {page['page_number']}: "
            f"{len(page['text'])} characters"
        )

    # 2. Test chunk generation
    chunks = create_document_chunks(pages)

    print("\n[2] CHUNK GENERATION")
    print("Total chunks:", len(chunks))

    # 3. Inspect individual chunks
    print("\n[3] CHUNK INSPECTION")

    for chunk in chunks:

        content = chunk["content"]

        print("\n" + "-" * 60)
        print("Chunk index :", chunk["chunk_index"])
        print("Page number :", chunk["page_number"])
        print("Characters  :", len(content))
        print("Content:")
        print(content)

    print("\n" + "=" * 60)
    print("INGESTION PIPELINE TEST COMPLETED")
    print("=" * 60)

    test_chunk_integrity()
    test_metadata_validation()
    test_embedding_validation()
    test_real_embedding_generation(chunks)


if __name__ == "__main__":
    main()
