
import argparse
import csv
import logging

from datetime import date
from pathlib import Path

from app.rag.database import get_connection


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================
# 1. CSV VALIDATION
# ==========================================

REQUIRED_FIELDS = {
    "source_key",
    "academic_year",
    "semester",
    "event_name",
    "event_category",
    "start_date",
    "end_date",
    "source_page",
    "source_text",
    "review_notes",
}


ALLOWED_SEMESTERS = {
    "Ganjil",
    "Genap",
    "Umum",
}


def validate_row(row, row_number):

    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if not row.get(field, "").strip()
    ]

    if missing_fields:
        raise ValueError(
            f"Row {row_number}: Missing fields: "
            f"{missing_fields}"
        )

    if row["semester"] not in ALLOWED_SEMESTERS:
        raise ValueError(
            f"Row {row_number}: Invalid semester."
        )

    start_date = date.fromisoformat(
        row["start_date"]
    )

    end_date = date.fromisoformat(
        row["end_date"]
    )

    if end_date < start_date:
        raise ValueError(
            f"Row {row_number}: End date "
            "cannot be earlier than start date."
        )

    source_page = int(row["source_page"])

    if source_page < 1:
        raise ValueError(
            f"Row {row_number}: Invalid source page."
        )

    return {
        "source_key": row["source_key"].strip(),
        "academic_year": row["academic_year"].strip(),
        "semester": row["semester"].strip(),
        "event_name": row["event_name"].strip(),
        "event_category": row["event_category"].strip(),
        "start_date": start_date,
        "end_date": end_date,
        "source_page": source_page,
        "source_text": row["source_text"].strip(),
        "review_notes": row["review_notes"].strip(),
    }


# ==========================================
# 2. READ CSV
# ==========================================

def load_calendar_csv(csv_path: Path):

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    events = []

    with csv_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        actual_fields = set(reader.fieldnames or [])

        missing_headers = REQUIRED_FIELDS - actual_fields

        if missing_headers:
            raise ValueError(
                f"CSV headers missing: {missing_headers}"
            )

        for row_number, row in enumerate(reader, start=2):

            if not row.get("source_key", "").strip():
                continue

            event = validate_row(
                row,
                row_number
            )

            events.append(event)

    if not events:
        raise ValueError(
            "No valid calendar events found."
        )

    return events


# ==========================================
# 3. IMPORT TO DATABASE
# ==========================================

def import_calendar_events(
    events,
    document_id: int
):

    inserted_count = 0
    skipped_count = 0

    with get_connection() as conn:

        with conn.cursor() as cur:

            # Verify source document
            cur.execute(
                """
                SELECT id, title, status
                FROM documents
                WHERE id = %s
                """,
                (document_id,)
            )

            document = cur.fetchone()

            if document is None:
                raise ValueError(
                    f"Document ID {document_id} not found."
                )

            logger.info(
                "Source document: %s",
                document[1]
            )

            if document[2] != "published":
                raise ValueError(
                    "Source document must be published "
                    "before its events can be imported."
                )

            for event in events:

                cur.execute(
                    """
                    INSERT INTO academic_events (
                        document_id,
                        source_key,
                        academic_year,
                        semester,
                        event_name,
                        event_category,
                        start_date,
                        end_date,
                        status,
                        source_page,
                        source_text,
                        review_notes
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, 'needs_review', %s, %s, %s
                    )
                    ON CONFLICT (
                        document_id,
                        source_key
                    )
                    DO NOTHING
                    RETURNING id
                    """,
                    (
                        document_id,
                        event["source_key"],
                        event["academic_year"],
                        event["semester"],
                        event["event_name"],
                        event["event_category"],
                        event["start_date"],
                        event["end_date"],
                        event["source_page"],
                        event["source_text"],
                        event["review_notes"],
                    )
                )

                inserted = cur.fetchone()

                if inserted:
                    inserted_count += 1

                    logger.info(
                        "Imported event ID: %s | %s",
                        inserted[0],
                        event["event_name"]
                    )

                else:
                    skipped_count += 1

                    logger.info(
                        "Skipped existing event: %s",
                        event["source_key"]
                    )

    return inserted_count, skipped_count


# ==========================================
# 4. MAIN
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="Structured Academic Calendar Ingestion"
    )

    parser.add_argument(
        "--file",
        required=True,
        help="Path to academic calendar CSV."
    )

    parser.add_argument(
        "--document-id",
        required=True,
        type=int,
        help="ID of source document in documents table."
    )

    args = parser.parse_args()

    try:

        csv_path = Path(args.file).resolve()

        events = load_calendar_csv(
            csv_path
        )

        logger.info(
            "Loaded %s calendar events.",
            len(events)
        )

        inserted, skipped = import_calendar_events(
            events=events,
            document_id=args.document_id
        )

        logger.info("=" * 50)

        logger.info(
            "CALENDAR INGESTION COMPLETED"
        )

        logger.info(
            "Total events       : %s",
            len(events)
        )

        logger.info(
            "Successfully added : %s",
            inserted
        )

        logger.info(
            "Skipped duplicates : %s",
            skipped
        )

        logger.info(
            "Initial status     : needs_review"
        )

    except Exception:

        logger.exception(
            "Calendar ingestion failed."
        )

        raise


if __name__ == "__main__":
    main()
