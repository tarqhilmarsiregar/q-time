
import re

from app.rag.config import DEFAULT_ACADEMIC_YEAR


TEMPORAL_PATTERNS = [
    r"\bkapan\b",
    r"\btanggal\b",
    r"\bjadwal\b",
    r"\bperiode\b",
    r"\bdimulai\b",
    r"\bmulai\b",
    r"\bberlangsung\b",
    r"\bdilaksanakan\b",
    r"\bberakhir\b",
    r"\bsampai\b",
    r"\bbatas waktu\b",
    r"\bdeadline\b",
    r"\bselesai\b",
    r"\bterakhir\b",
]

KNOWLEDGE_PATTERNS = [
    r"\bapa itu\b",
    r"\bapa yang dimaksud\b",
    r"\bpengertian\b",
    r"\bdefinisi\b",
    r"\bjelaskan\b",
    r"\bpenjelasan\b",
    r"\bsyarat\b",
    r"\bpersyaratan\b",
    r"\bprosedur\b",
    r"\btata cara\b",
    r"\bmekanisme\b",
    r"\bketentuan\b",
    r"\bbagaimana cara\b",
    r"\bcara mengajukan\b",
    r"\bbiaya\b",
    r"\bberapa biaya\b",
]

CALENDAR_CONTEXT_PATTERNS = [
    r"\bkalender akademik\b",
    r"\bsemester\b",
    r"\buts\b",
    r"\buas\b",
    r"\bkrs\b",
    r"\bkhs\b",
    r"\bminggu tenang\b",
    r"\blibur semester\b",
    r"\bperkuliahan\b",
    r"\bujian\b",
]


def normalize_query(query: str) -> str:
    """Normalize user query for intent and entity detection."""
    return re.sub(r"\s+", " ", query.strip().lower())


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def detect_academic_year(query: str) -> str:
    """
    Detect academic year in formats such as:
    2026/2027 or 2026-2027.
    """
    match = re.search(
        r"\b(20\d{2})\s*[/\-]\s*(20\d{2})\b",
        query,
    )

    if match:
        return f"{match.group(1)}/{match.group(2)}"

    return DEFAULT_ACADEMIC_YEAR


def detect_semester(query: str) -> str | None:
    if re.search(r"\bganjil\b", query):
        return "Ganjil"

    if re.search(r"\bgenap\b", query):
        return "Genap"

    return None


def is_knowledge_question(query: str) -> bool:
    return _matches_any(query, KNOWLEDGE_PATTERNS)


def has_temporal_intent(query: str) -> bool:
    return _matches_any(query, TEMPORAL_PATTERNS)


def detect_calendar_event(query: str) -> dict | None:
    """
    Match user expressions to actual event names stored
    in academic_events.

    Keyword values must match the event_name content
    in the database.
    """

    # UTS susulan is not currently recorded in our calendar data.
    # Keep a specific keyword so it does not accidentally return
    # all examination events.
    if re.search(
        r"\buts\s+susulan\b"
        r"|\bsusulan\s+uts\b"
        r"|\bujian\s+susulan\s+tengah\s+semester\b",
        query,
    ):
        return {
            "keyword": "Ujian Susulan Tengah Semester",
            "category": "exam",
        }

    if re.search(
        r"\buas\s+susulan\b"
        r"|\bsusulan\s+uas\b"
        r"|\bujian\s+susulan\s+akhir\s+semester\b",
        query,
    ):
        return {
            "keyword": "Ujian Susulan Akhir Semester",
            "category": "exam",
        }

    if (
        "minggu tenang" in query
        and re.search(r"\buas\b", query)
    ):
        return {
            "keyword": "Minggu Tenang UAS",
            "category": "academic_break",
        }

    if (
        "minggu tenang" in query
        and re.search(r"\buts\b", query)
    ):
        return {
            "keyword": "Minggu Tenang UTS",
            "category": "academic_break",
        }

    if "minggu tenang" in query:
        return {
            "keyword": "Minggu Tenang",
            "category": "academic_break",
        }

    if re.search(r"\bkrs\b", query):
        return {
            "keyword": "Pengisian KRS",
            "category": "registration",
        }

    if re.search(
        r"\bperkuliahan perdana\b"
        r"|\bawal perkuliahan\b"
        r"|\bkuliah dimulai\b"
        r"|\bmulai perkuliahan\b"
        r"|\bsemester dimulai\b",
        query,
    ):
        return {
            "keyword": "Perkuliahan Perdana",
            "category": "lecture",
        }

    if re.search(r"\blibur semester\b", query):
        return {
            "keyword": "Libur Semester",
            "category": "academic_break",
        }

    if re.search(r"\buts\b", query):
        return {
            "keyword": "Ujian Tengah Semester",
            "category": "exam",
        }

    if re.search(r"\buas\b", query):
        return {
            "keyword": "Ujian Akhir Semester",
            "category": "exam",
        }

    return None


def classify_query(query: str) -> dict:
    """
    Routes a question to:
    - structured: academic calendar event retrieval
    - semantic: knowledge/document retrieval

    Knowledge intent takes priority unless the user also
    explicitly asks about time, date, or schedule.
    """

    normalized = normalize_query(query)

    academic_year = detect_academic_year(normalized)
    semester = detect_semester(normalized)

    knowledge_intent = is_knowledge_question(normalized)
    temporal_intent = has_temporal_intent(normalized)

    calendar_context = _matches_any(
        normalized,
        CALENDAR_CONTEXT_PATTERNS,
    )

    event = detect_calendar_event(normalized)

    # Example:
    # "Apa itu UAS?" -> semantic
    # "Apa persyaratan pengajuan cuti?" -> semantic
    if knowledge_intent and not temporal_intent:
        return {
            "route": "semantic",
            "reason": "Knowledge-oriented question",
            "academic_year": academic_year,
            "semester": semester,
            "event_keyword": None,
            "event_category": None,
        }

    # Example:
    # "Kapan UAS Ganjil?"
    # "Jadwal KRS Genap?"
    # "Sampai kapan libur semester Genap?"
    if calendar_context and (temporal_intent or event is not None):
        return {
            "route": "structured",
            "reason": "Academic calendar question",
            "academic_year": academic_year,
            "semester": semester,
            "event_keyword": event["keyword"] if event else None,
            "event_category": event["category"] if event else None,
        }

    # Explicit event detected, even if the wording does not
    # contain "kapan" or "jadwal".
    if event is not None and not knowledge_intent:
        return {
            "route": "structured",
            "reason": "Academic event detected",
            "academic_year": academic_year,
            "semester": semester,
            "event_keyword": event["keyword"],
            "event_category": event["category"],
        }

    return {
        "route": "semantic",
        "reason": "General knowledge retrieval",
        "academic_year": academic_year,
        "semester": semester,
        "event_keyword": None,
        "event_category": None,
    }
