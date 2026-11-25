
from functools import lru_cache

import numpy as np

from sentence_transformers import SentenceTransformer

from app.rag.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDING_BATCH_SIZE,
)


@lru_cache(maxsize=1)
def load_embedding_model():
    """
    Load embedding model once and cache it.
    CPU is used for local development.
    """

    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu"
    )

    return model


def embed_passages(texts: list[str]) -> np.ndarray:
    """
    Generate document embeddings.

    E5 requires the 'passage:' prefix
    for document retrieval tasks.
    """

    if not texts:
        return np.empty((0, EMBEDDING_DIMENSION))

    model = load_embedding_model()

    inputs = [
        f"passage: {text.strip()}"
        for text in texts
    ]

    embeddings = model.encode(
        inputs,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    if embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            "Unexpected embedding dimension: "
            f"{embeddings.shape[1]}"
        )

    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Generate embedding for a user query.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    model = load_embedding_model()

    embedding = model.encode(
        [f"query: {query.strip()}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    if embedding.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            "Unexpected query embedding dimension."
        )

    return embedding[0]
