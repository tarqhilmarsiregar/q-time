
from pgvector import Vector

from app.rag.database import get_connection
from app.rag.embedding import embed_query
from app.rag.config import TOP_K


# ==========================================
# 1. SEMANTIC RETRIEVAL ENGINE
# ==========================================

def retrieve_documents(
    query: str,
    top_k=TOP_K,
    category=None,
    academic_year=None,
    exclude_categories=None,
):
    """
    Retrieve published academic document chunks
    using cosine similarity through pgvector.

    Parameters
    ----------
    query : str
        User question.

    top_k : int
        Maximum number of retrieved chunks.

    category : str | None
        Optional document category filter.

    academic_year : str | None
        Optional academic year filter.

    exclude_categories : list[str] | None
        Document categories excluded from retrieval.

    Returns
    -------
    list[dict]
        Ranked document chunks and metadata.
    """

    # --------------------------------------
    # Validate input
    # --------------------------------------

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not isinstance(top_k, int) or not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20.")

    # --------------------------------------
    # Generate query embedding
    # --------------------------------------

    query_vector = embed_query(query)

    vector = Vector(query_vector.tolist())

    # --------------------------------------
    # Build SQL filters safely
    # --------------------------------------

    conditions = [
        "d.status = %s"
    ]

    # First vector belongs to SELECT distance.
    params = [vector, "published"]

    # Optional category filter
    if category:
        conditions.append("d.category = %s")
        params.append(category)

    # Optional academic year filter
    if academic_year:
        conditions.append("d.academic_year = %s")
        params.append(academic_year)

    # --------------------------------------
    # RETRIEVAL GOVERNANCE
    # Exclude restricted document categories
    # --------------------------------------

    if isinstance(exclude_categories, str):
        excluded = (
            [exclude_categories.strip().lower()]
            if exclude_categories.strip()
            else []
        )
    else:
        excluded = [
            value.strip().lower()
            for value in (exclude_categories or [])
            if isinstance(value, str) and value.strip()
        ]

    if excluded:
        placeholders = ", ".join(
            ["%s"] * len(excluded)
        )

        conditions.append(
            "LOWER(COALESCE(d.category, '')) "
            f"NOT IN ({placeholders})"
        )

        params.extend(excluded)

    # Combine all WHERE conditions
    where_clause = " AND ".join(conditions)

    # --------------------------------------
    # Semantic search query
    # --------------------------------------

    sql = f"""
        SELECT
            dc.id AS chunk_id,
            d.id AS document_id,
            d.title,
            d.file_name,
            d.category,
            d.academic_year,
            d.version,
            dc.chunk_index,
            dc.page_number,
            dc.content,

            (dc.embedding <=> %s) AS cosine_distance

        FROM document_chunks dc

        INNER JOIN documents d
            ON d.id = dc.document_id

        WHERE {where_clause}

        ORDER BY dc.embedding <=> %s

        LIMIT %s
    """

    # Vector for ORDER BY and top_k limit
    params.extend([
        vector,
        top_k
    ])

    # --------------------------------------
    # Execute database query
    # --------------------------------------

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(sql, params)

            rows = cur.fetchall()

    # --------------------------------------
    # Format retrieval results
    # --------------------------------------

    results = []

    for row in rows:

        distance = float(row[10])

        result = {
            "chunk_id": row[0],
            "document_id": row[1],
            "title": row[2],
            "file_name": row[3],
            "category": row[4],
            "academic_year": row[5],
            "version": row[6],
            "chunk_index": row[7],
            "page_number": row[8],
            "content": row[9],

            "cosine_distance": distance,

            # Ranking score, not probability of correctness.
            "similarity": 1.0 - distance
        }

        results.append(result)

    return results


# ==========================================
# 2. DISPLAY RETRIEVAL RESULTS
# ==========================================

def display_results(results):

    if not results:
        print("\nTidak ditemukan dokumen yang relevan.")
        return

    print("\n" + "=" * 65)
    print("SEMANTIC RETRIEVAL RESULTS")
    print("=" * 65)

    for index, result in enumerate(results, start=1):

        print("\n" + "-" * 65)

        print(f"Rank           : {index}")
        print(f"Chunk ID       : {result['chunk_id']}")
        print(f"Document ID    : {result['document_id']}")
        print(f"Title          : {result['title']}")
        print(f"File           : {result['file_name']}")
        print(f"Category       : {result['category']}")
        print(f"Academic Year  : {result['academic_year']}")
        print(f"Version        : {result['version']}")
        print(f"Chunk Index    : {result['chunk_index']}")
        print(f"Page Number    : {result['page_number']}")

        print(
            f"Cosine Distance: "
            f"{result['cosine_distance']:.6f}"
        )

        print(
            f"Similarity     : "
            f"{result['similarity']:.6f}"
        )

        print("\nCONTENT:")
        print(result["content"])

    print("\n" + "=" * 65)


# ==========================================
# 3. INTERACTIVE CLI TESTING
# ==========================================

def main():

    print("\nSTMIK ACADEMIC COPILOT")
    print("SEMANTIC RETRIEVAL TESTING")
    print("=" * 65)

    print("\nDocument category filter:")
    print("1. academic-calendar")
    print("2. academic-regulation")
    print("3. All categories")

    category_choice = input(
        "\nPilih kategori (1/2/3): "
    ).strip()

    category_mapping = {
        "1": "academic-calendar",
        "2": "academic-regulation",
        "3": None
    }

    if category_choice not in category_mapping:
        print("Pilihan kategori tidak valid.")
        return

    selected_category = category_mapping[category_choice]

    academic_year = None

    if selected_category == "academic-calendar":

        academic_year = input(
            "Masukkan tahun akademik (kosong = semua): "
        ).strip() or None

    print("\nSilakan masukkan pertanyaan.")
    print("Ketik 'exit' untuk berhenti.")

    while True:

        print("\n" + "-" * 65)

        query = input("Pertanyaan: ").strip()

        if query.casefold() in ["exit", "quit"]:
            print("\nRetrieval testing selesai.")
            break

        if not query:
            print("Pertanyaan tidak boleh kosong.")
            continue

        try:

            results = retrieve_documents(
                query=query,
                top_k=5,
                category=selected_category,
                academic_year=academic_year
            )

            display_results(results)

        except Exception as error:

            print("\nTerjadi kesalahan saat retrieval:")
            print(str(error))


if __name__ == "__main__":
    main()
