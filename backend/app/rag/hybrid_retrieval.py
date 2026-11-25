
import argparse
import logging

from app.rag.query_router import classify_query

from app.rag.calendar_retrieval import (
    retrieve_academic_events,
    format_date_range,
)

from app.rag.retrieval import retrieve_documents


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================
# 1. HYBRID RETRIEVAL ENGINE
# ==========================================

def hybrid_retrieve(
    query: str,
    include_review: bool = False,
    top_k: int = 5,
    academic_year_override: str | None = None
):
    """
    Route a user query to either:

    1. Structured SQL retrieval.
    2. Semantic vector retrieval.

    The router currently uses deterministic rules.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20.")

    # --------------------------------------
    # Query classification
    # --------------------------------------

    route_info = classify_query(query)

    academic_year = (
        academic_year_override
        or route_info["academic_year"]
    )

    logger.info(
        "Query route: %s",
        route_info["route"]
    )

    logger.info(
        "Routing reason: %s",
        route_info["reason"]
    )

    # --------------------------------------
    # STRUCTURED RETRIEVAL
    # --------------------------------------

    if route_info["route"] == "structured":

        event = route_info["event"]

        results = retrieve_academic_events(
            academic_year=academic_year,

            event_keyword=(
                event["keyword"]
                if event else None
            ),

            semester=route_info["semester"],

            event_category=(
                event["category"]
                if event else None
            ),

            include_review=include_review,

            limit=(
                20 if event is None else top_k
            )
        )

        return {
            "route": "structured",
            "query": query,
            "reason": route_info["reason"],
            "academic_year": academic_year,
            "semester": route_info["semester"],
            "event": event,
            "results": results
        }

    # --------------------------------------
    # SEMANTIC RETRIEVAL
    # --------------------------------------

    results = retrieve_documents(
        query=query,
        top_k=top_k,
        category=None
    )

    return {
        "route": "semantic",
        "query": query,
        "reason": route_info["reason"],
        "academic_year": academic_year,
        "semester": route_info["semester"],
        "event": None,
        "results": results
    }


# ==========================================
# 2. DISPLAY STRUCTURED RESULTS
# ==========================================

def display_structured_results(results):

    if not results:
        print("\nTidak ditemukan kegiatan akademik.")
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
# 3. DISPLAY SEMANTIC RESULTS
# ==========================================

def display_semantic_results(results):

    if not results:
        print("\nTidak ditemukan potongan dokumen.")
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
            f"Similarity : "
            f"{result['similarity']:.6f}"
        )

        print("\nCONTENT:")
        print(result["content"])

    print("\n" + "=" * 65)
    print(f"Total chunks found: {len(results)}")


# ==========================================
# 4. RESULT DISPLAY
# ==========================================

def display_hybrid_result(response):

    print("\n" + "=" * 65)
    print("STMIK ACADEMIC COPILOT")
    print("HYBRID RETRIEVAL ENGINE")
    print("=" * 65)

    print(f"\nQuestion : {response['query']}")

    print(f"Route    : {response['route']}")

    print(f"Reason   : {response['reason']}")

    if response["academic_year"]:
        print(
            f"Academic Year : {response['academic_year']}"
        )

    if response["semester"]:
        print(
            f"Semester      : {response['semester']}"
        )

    if response["route"] == "structured":

        display_structured_results(
            response["results"]
        )

    else:

        display_semantic_results(
            response["results"]
        )


# ==========================================
# 5. CLI TESTING
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

    # Single-query mode
    if args.query:

        run_query(args.query)

        return

    # Interactive mode
    print("\nSTMIK ACADEMIC COPILOT")
    print("HYBRID RETRIEVAL TESTING")

    print("\nMode:")
    print(
        "PREVIEW - includes needs_review"
        if args.preview
        else "PUBLISHED ONLY"
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
