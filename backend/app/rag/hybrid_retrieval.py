
import argparse
import logging

from app.rag.query_router import classify_query

from app.rag.calendar_retrieval import (
    retrieve_academic_events,
    format_date_range,
)

from app.rag.retrieval import retrieve_documents

from app.rag.config import (
    SEMANTIC_EXCLUDED_CATEGORIES,
    SEMANTIC_MIN_SIMILARITY,
)


# ==========================================
# LOGGING CONFIGURATION
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================
# 1. SEMANTIC RETRIEVAL GOVERNANCE
# ==========================================

def filter_trusted_semantic_results(documents):
    """
    Apply application-level governance to semantic results.

    Rules:
    1. Exclude restricted document categories.
    2. Reject results without a valid similarity score.
    3. Reject results below the configured threshold.

    Published document filtering is enforced separately
    inside retrieve_documents().
    """

    trusted_documents = []

    excluded_categories = {
        category.strip().lower()
        for category in SEMANTIC_EXCLUDED_CATEGORIES
        if isinstance(category, str) and category.strip()
    }

    for document in documents:

        # ----------------------------------
        # Check document category
        # ----------------------------------

        category = str(
            document.get("category") or ""
        ).strip().lower()

        if category in excluded_categories:
            logger.warning(
                "Excluded restricted category: %s",
                category
            )
            continue

        # ----------------------------------
        # Check similarity score
        # ----------------------------------

        try:
            similarity = float(
                document["similarity"]
            )
        except (KeyError, TypeError, ValueError):
            logger.warning(
                "Document rejected: invalid similarity score."
            )
            continue

        if similarity < SEMANTIC_MIN_SIMILARITY:
            logger.info(
                "Document rejected: similarity %.6f is below %.6f",
                similarity,
                SEMANTIC_MIN_SIMILARITY
            )
            continue

        trusted_documents.append(document)

    return trusted_documents


# ==========================================
# 2. HYBRID RETRIEVAL ENGINE
# ==========================================

def hybrid_retrieve(
    query: str,
    include_review: bool = False,
    top_k: int = 5,
    academic_year_override: str | None = None
):
    """
    Route an academic question to either:

    1. Structured SQL retrieval.
    2. Semantic vector retrieval.

    Governance:
    - Preview only affects structured calendar retrieval.
    - Semantic retrieval only uses published documents.
    - Restricted categories are excluded.
    - Similarity threshold is enforced.
    """

    # --------------------------------------
    # Validate input
    # --------------------------------------

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not isinstance(top_k, int) or not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20.")

    # --------------------------------------
    # Query classification
    # --------------------------------------

    route_info = classify_query(query)

    route = route_info["route"]

    if route not in ("structured", "semantic"):
        raise ValueError(
            f"Unsupported query route: {route}"
        )

    academic_year = (
        academic_year_override
        or route_info.get("academic_year")
    )

    semester = route_info.get("semester")

    # --------------------------------------
    # Normalize event information
    # --------------------------------------

    # Current query_router.py returns:
    # event_keyword and event_category.

    event_keyword = route_info.get("event_keyword")
    event_category = route_info.get("event_category")

    # Backward compatibility:
    # Support older router versions that return
    # an event dictionary.

    legacy_event = route_info.get("event")

    if isinstance(legacy_event, dict):

        if event_keyword is None:
            event_keyword = legacy_event.get("keyword")

        if event_category is None:
            event_category = legacy_event.get("category")

    event = None

    if event_keyword or event_category:

        event = {
            "keyword": event_keyword,
            "category": event_category
        }

    logger.info("Query route: %s", route)
    logger.info(
        "Routing reason: %s",
        route_info.get("reason", "Not specified")
    )

    # ======================================
    # STRUCTURED RETRIEVAL
    # ======================================

    if route == "structured":

        logger.info(
            "Structured retrieval preview: %s",
            include_review
        )

        results = retrieve_academic_events(
            academic_year=academic_year,

            event_keyword=event_keyword,

            semester=semester,

            event_category=event_category,

            include_review=include_review,

            limit=(
                20 if event_keyword is None else top_k
            )
        )

        return {
            "route": "structured",
            "query": query,
            "reason": route_info.get("reason"),
            "academic_year": academic_year,
            "semester": semester,
            "event": event,
            "results": results,
            "preview": include_review
        }

    # ======================================
    # SEMANTIC RETRIEVAL
    # ======================================

    logger.info(
        "Semantic minimum similarity: %.2f",
        SEMANTIC_MIN_SIMILARITY
    )

    logger.info(
        "Excluded document categories: %s",
        SEMANTIC_EXCLUDED_CATEGORIES
    )

    # Retrieve only published documents.
    # Category exclusions are applied in SQL
    # through retrieve_documents().

    documents = retrieve_documents(
        query=query,
        top_k=top_k,
        category=None,
        academic_year=None,
        exclude_categories=SEMANTIC_EXCLUDED_CATEGORIES
    )

    # Apply an additional application-level filter.

    trusted_documents = filter_trusted_semantic_results(
        documents
    )

    logger.info(
        "Semantic chunks retrieved: %d",
        len(documents)
    )

    logger.info(
        "Trusted semantic chunks: %d",
        len(trusted_documents)
    )

    return {
        "route": "semantic",
        "query": query,
        "reason": route_info.get("reason"),
        "academic_year": academic_year,
        "semester": semester,
        "event": None,
        "results": trusted_documents,
        "preview": False,
        "semantic_min_similarity": SEMANTIC_MIN_SIMILARITY
    }


# ==========================================
# 3. DISPLAY STRUCTURED RESULTS
# ==========================================

def display_structured_results(results, preview=False):

    if not results:

        print("\nTidak ditemukan kegiatan akademik.")

        if not preview:
            print(
                "Data yang belum berstatus published "
                "tidak ditampilkan."
            )

        return

    print("\nSTRUCTURED RETRIEVAL RESULTS")

    for index, event in enumerate(results, start=1):

        date_range = format_date_range(
            event["start_date"],
            event["end_date"]
        )

        print("\n" + "-" * 65)

        print(f"Rank          : {index}")
        print(f"Event         : {event['event_name']}")
        print(f"Academic Year : {event['academic_year']}")
        print(f"Semester      : {event['semester']}")
        print(f"Category      : {event['event_category']}")
        print(f"Period        : {date_range}")
        print(f"Status        : {event['status']}")
        print(f"Source        : {event['source_document']}")
        print(f"Source Page   : {event['source_page']}")

    print("\n" + "=" * 65)
    print(f"Total events found: {len(results)}")


# ==========================================
# 4. DISPLAY SEMANTIC RESULTS
# ==========================================

def display_semantic_results(results):

    if not results:

        print(
            "\nTidak ditemukan sumber pengetahuan "
            "yang terverifikasi dan cukup relevan."
        )

        print(
            "Sistem tidak akan menyusun jawaban "
            "berdasarkan sumber yang tidak tersedia."
        )

        return

    print("\nSEMANTIC RETRIEVAL RESULTS")

    for index, result in enumerate(results, start=1):

        print("\n" + "-" * 65)

        print(f"Rank       : {index}")
        print(f"Document   : {result['title']}")
        print(f"Category   : {result['category']}")
        print(f"Version    : {result['version']}")
        print(f"Page       : {result['page_number']}")

        print(
            f"Similarity : {result['similarity']:.6f}"
        )

        print("\nCONTENT:")
        print(result["content"])

    print("\n" + "=" * 65)
    print(f"Total trusted chunks: {len(results)}")


# ==========================================
# 5. RESULT DISPLAY
# ==========================================

def display_hybrid_result(response):

    print("\n" + "=" * 65)
    print("STMIK ACADEMIC COPILOT")
    print("HYBRID RETRIEVAL ENGINE")
    print("=" * 65)

    print(f"\nQuestion : {response['query']}")
    print(f"Route    : {response['route']}")
    print(f"Reason   : {response['reason']}")

    if response.get("academic_year"):
        print(
            f"Academic Year : {response['academic_year']}"
        )

    if response.get("semester"):
        print(
            f"Semester      : {response['semester']}"
        )

    if response["route"] == "structured":

        print(
            "Mode          : "
            + (
                "PREVIEW"
                if response["preview"]
                else "PUBLISHED ONLY"
            )
        )

        display_structured_results(
            response["results"],
            preview=response["preview"]
        )

    else:

        print("Mode          : PUBLISHED DOCUMENTS ONLY")

        print(
            "Minimum Similarity: "
            f"{response['semantic_min_similarity']:.2f}"
        )

        display_semantic_results(
            response["results"]
        )


# ==========================================
# 6. CLI TESTING
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="STMIK Academic Copilot Hybrid Retrieval"
    )

    parser.add_argument(
        "--query",
        default=None,
        help="Single question to test."
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Include needs_review calendar events."
    )

    parser.add_argument(
        "--academic-year",
        default=None,
        help="Override default academic year."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5
    )

    args = parser.parse_args()

    if not 1 <= args.top_k <= 20:
        parser.error("--top-k must be between 1 and 20.")

    def run_query(query):

        try:

            response = hybrid_retrieve(
                query=query,
                include_review=args.preview,
                top_k=args.top_k,
                academic_year_override=args.academic_year
            )

            display_hybrid_result(response)

        except Exception:

            logger.exception(
                "Hybrid retrieval failed."
            )

    # --------------------------------------
    # Single-query mode
    # --------------------------------------

    if args.query:

        run_query(args.query)
        return

    # --------------------------------------
    # Interactive mode
    # --------------------------------------

    print("\nSTMIK ACADEMIC COPILOT")
    print("HYBRID RETRIEVAL TESTING")

    print("\nPreview setting:")
    print(
        "ON - affects Structured Retrieval only"
        if args.preview
        else "OFF - published events only"
    )

    print(
        "Semantic Retrieval always uses published "
        "documents and governance filters."
    )

    print("\nKetik 'exit' untuk berhenti.")

    while True:

        print("\n" + "-" * 65)

        query = input(
            "Pertanyaan mahasiswa: "
        ).strip()

        if query.casefold() in ["exit", "quit"]:

            print("\nTesting selesai.")
            break

        if not query:

            print("Pertanyaan tidak boleh kosong.")
            continue

        run_query(query)


if __name__ == "__main__":
    main()
