
import argparse
import logging

from datetime import date

from psycopg.rows import dict_row

from app.rag.database import get_connection


# ==========================================
# 1. LOGGING
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================
# 2. DATE FORMATTER
# ==========================================

INDONESIAN_MONTHS = [
    "Januari",
    "Februari",
    "Maret",
    "April",
    "Mei",
    "Juni",
    "Juli",
    "Agustus",
    "September",
    "Oktober",
    "November",
    "Desember"
]


def format_date(value: date) -> str:
    """
    Format Python date into Indonesian date.
    """

    return (
        f"{value.day} "
        f"{INDONESIAN_MONTHS[value.month - 1]} "
        f"{value.year}"
    )


def format_date_range(start_date: date, end_date: date) -> str:
    """
    Format date range for academic events.
    """

    if start_date == end_date:
        return format_date(start_date)

    if (
        start_date.month == end_date.month
        and start_date.year == end_date.year
    ):
        return (
            f"{start_date.day}–{end_date.day} "
            f"{INDONESIAN_MONTHS[start_date.month - 1]} "
            f"{start_date.year}"
        )

    return (
        f"{format_date(start_date)} – "
        f"{format_date(end_date)}"
    )


# ==========================================
# 3. STRUCTURED CALENDAR RETRIEVAL
# ==========================================

def retrieve_academic_events(
    academic_year: str,
    event_keyword: str | None = None,
    semester: str | None = None,
    event_category: str | None = None,
    include_review: bool = False,
    limit: int = 20
):
    """
    Retrieve structured academic calendar events.

    Default:
        Only published events.

    Preview:
        Published + needs_review events.

    Parameters
    ----------
    academic_year : str
        Academic year, e.g. 2026/2027.

    event_keyword : str | None
        Search keyword in event name.

    semester : str | None
        Ganjil, Genap, or Umum.

    event_category : str | None
        exam, registration, lecture, academic_break, etc.

    include_review : bool
        Include needs_review events for internal testing.

    limit : int
        Maximum number of records.
    """

    if not academic_year or not academic_year.strip():
        raise ValueError("Academic year is required.")

    if limit < 1 or limit > 100:
        raise ValueError("Limit must be between 1 and 100.")

    if semester and semester not in {
        "Ganjil",
        "Genap",
        "Umum"
    }:
        raise ValueError("Invalid semester.")

    if event_keyword and not event_keyword.strip():
        event_keyword = None

    conditions = [
        "e.academic_year = %s",
        "d.status = 'published'"
    ]

    params = [academic_year]

    # Status filtering
    if include_review:
        conditions.append(
            "e.status IN ('published', 'needs_review')"
        )
    else:
        conditions.append(
            "e.status = 'published'"
        )

    # Event name filtering
    if event_keyword:
        conditions.append(
            "e.event_name ILIKE %s"
        )

        params.append(
            f"%{event_keyword.strip()}%"
        )

    # Semester filtering
    if semester:
        conditions.append(
            "e.semester = %s"
        )

        params.append(semester)

    # Category filtering
    if event_category:
        conditions.append(
            "e.event_category = %s"
        )

        params.append(event_category)

    where_clause = " AND ".join(conditions)

    sql = f"""
        SELECT
            e.id,
            e.source_key,
            e.event_name,
            e.academic_year,
            e.semester,
            e.event_category,
            e.start_date,
            e.end_date,
            e.status,
            e.source_page,
            e.source_text,
            e.review_notes,
            e.verified_by,
            e.verified_at,

            d.title AS source_document,
            d.file_name AS source_file

        FROM academic_events e

        INNER JOIN documents d
            ON d.id = e.document_id

        WHERE {where_clause}

        ORDER BY
            e.start_date ASC,
            e.id ASC

        LIMIT %s
    """

    params.append(limit)

    # Execute query
    with get_connection() as conn:

        with conn.cursor(
            row_factory=dict_row
        ) as cur:

            cur.execute(
                sql,
                params
            )

            results = cur.fetchall()

    return results


# ==========================================
# 4. DISPLAY RESULTS
# ==========================================

def display_events(results):

    if not results:

        print("\nTidak ditemukan kegiatan akademik.")

        return

    print("\n" + "=" * 70)
    print("STMIK ACADEMIC COPILOT")
    print("STRUCTURED CALENDAR RETRIEVAL")
    print("=" * 70)

    for index, event in enumerate(results, start=1):

        date_range = format_date_range(
            event["start_date"],
            event["end_date"]
        )

        print("\n" + "-" * 70)

        print(f"Rank          : {index}")
        print(f"Event ID      : {event['id']}")
        print(f"Event Name    : {event['event_name']}")
        print(f"Academic Year : {event['academic_year']}")
        print(f"Semester      : {event['semester']}")
        print(f"Category      : {event['event_category']}")

        print(f"Date          : {date_range}")

        print(f"Status        : {event['status']}")

        print(f"Source        : {event['source_document']}")
        print(f"Source Page   : {event['source_page']}")

        if event["review_notes"]:
            print(
                f"Review Notes  : {event['review_notes']}"
            )

    print("\n" + "=" * 70)

    print(
        f"Total events found: {len(results)}"
    )


# ==========================================
# 5. INTERACTIVE CLI
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="Academic Calendar Structured Retrieval"
    )

    parser.add_argument(
        "--academic-year",
        default="2026/2027"
    )

    parser.add_argument(
        "--keyword",
        default=None
    )

    parser.add_argument(
        "--semester",
        choices=["Ganjil", "Genap", "Umum"],
        default=None
    )

    parser.add_argument(
        "--category",
        default=None
    )

    parser.add_argument(
        "--include-review",
        action="store_true",
        help="Include needs_review events for development testing."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20
    )

    args = parser.parse_args()

    try:

        results = retrieve_academic_events(
            academic_year=args.academic_year,
            event_keyword=args.keyword,
            semester=args.semester,
            event_category=args.category,
            include_review=args.include_review,
            limit=args.limit
        )

        display_events(results)

    except Exception:

        logger.exception(
            "Calendar retrieval failed."
        )

        raise


if __name__ == "__main__":
    main()
