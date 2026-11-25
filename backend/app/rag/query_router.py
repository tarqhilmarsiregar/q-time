
import re

from app.rag.config import DEFAULT_ACADEMIC_YEAR


# ==========================================
# 1. QUERY NORMALIZATION
# ==========================================

def normalize_query(query: str) -> str:

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    return " ".join(query.casefold().split())


# ==========================================
# 2. ACADEMIC YEAR DETECTION
# ==========================================

def detect_academic_year(query: str) -> str:

    pattern = (
        r"\b(20\d{2})\s*[/\-]\s*(20\d{2})\b"
    )

    match = re.search(pattern, query)

    if match:
        return f"{match.group(1)}/{match.group(2)}"

    return DEFAULT_ACADEMIC_YEAR


# ==========================================
# 3. SEMESTER DETECTION
# ==========================================

def detect_semester(query: str) -> str | None:

    if re.search(r"\bganjil\b", query):
        return "Ganjil"

    if re.search(r"\bgenap\b", query):
        return "Genap"

    return None


# ==========================================
# 4. PROCEDURAL QUESTION DETECTION
# ==========================================

def is_knowledge_question(query: str) -> bool:

    knowledge_patterns = [
        r"\bapa\s+itu\b",
        r"\bpengertian\b",
        r"\bsyarat\b",
        r"\bpersyaratan\b",
        r"\bprosedur\b",
        r"\btata\s+cara\b",
        r"\bbagaimana\s+cara\b",
        r"\bketentuan\b",
        r"\bmekanisme\b",
        r"\bpenjelasan\b",
    ]

    return any(
        re.search(pattern, query)
        for pattern in knowledge_patterns
    )


# ==========================================
# 5. CALENDAR EVENT DETECTION
# ==========================================

def detect_calendar_event(query: str):

    # More specific events must be checked first.

    # UAS / UTS supplementary exams
    if (
        "susulan" in query
        and (
            "uas" in query
            or "ujian akhir" in query
            or "uts" in query
            or "ujian tengah" in query
        )
    ):

        if "uts" in query or "ujian tengah" in query:
            return {
            "keyword": "Ujian Susulan Tengah Semester",
            "category": "exam",
            "description": "Ujian Susulan UTS",
        }

        return {
            "keyword": "Ujian Susulan Akhir Semester",
            "category": "exam",
            "description": "UAS Susulan",
        }

    # Quiet week before UAS
    if (
        "minggu tenang" in query
        and (
            "uas" in query
            or "ujian akhir" in query
        )
    ):

        return {
            "keyword": "Minggu Tenang Menyambut UAS",
            "category": "academic_break",
            "description": "Minggu Tenang UAS",
        }

    # Quiet week before UTS
    if (
        "minggu tenang" in query
        and (
            "uts" in query
            or "ujian tengah" in query
        )
    ):

        return {
            "keyword": "Minggu Tenang UTS",
            "category": "academic_break",
            "description": "Minggu Tenang UTS",
        }

    # Generic quiet week
    if "minggu tenang" in query:

        return {
            "keyword": "Minggu Tenang",
            "category": "academic_break",
            "description": "Minggu Tenang",
        }

    # Midterm examination
    if (
        re.search(r"\buts\b", query)
        or "ujian tengah semester" in query
    ):

        return {
            "keyword": "Ujian Tengah Semester",
            "category": "exam",
            "description": "UTS",
        }

    # Final examination
    if (
        re.search(r"\buas\b", query)
        or "ujian akhir semester" in query
    ):

        return {
            "keyword": "Ujian Akhir Semester",
            "category": "exam",
            "description": "UAS",
        }

    # Registration / KRS
    if (
        re.search(r"\bkrs\b", query)
        or "pengisian krs" in query
    ):

        return {
            "keyword": "Pengisian KRS",
            "category": "registration",
            "description": "KRS",
        }

    # First lecture
    if (
        "perkuliahan perdana" in query
        or "kuliah dimulai" in query
        or "awal perkuliahan" in query
    ):

        return {
            "keyword": "Perkuliahan Perdana",
            "category": "lecture",
            "description": "Perkuliahan Perdana",
        }

    # Semester break
    if "libur semester" in query:

        return {
            "keyword": "Libur Semester",
            "category": "academic_break",
            "description": "Libur Semester",
        }

    return None


# ==========================================
# 6. MAIN QUERY ROUTER
# ==========================================

def classify_query(query: str) -> dict:

    normalized = normalize_query(query)

    academic_year = detect_academic_year(
        normalized
    )

    semester = detect_semester(
        normalized
    )

    # Procedural and explanatory questions
    # should go to semantic retrieval.
    if is_knowledge_question(normalized):

        return {
            "route": "semantic",
            "query": query,
            "academic_year": academic_year,
            "semester": semester,
            "event": None,
            "reason": "Knowledge-oriented question"
        }

    event = detect_calendar_event(
        normalized
    )

    calendar_keywords = [
        "jadwal",
        "kapan",
        "tanggal",
        "periode",
        "kalender akademik",
        "dilaksanakan",
        "dimulai",
        "berlangsung",
    ]

    is_calendar_question = any(
        keyword in normalized
        for keyword in calendar_keywords
    )

    # Calendar event queries
    if event is not None or is_calendar_question:

        return {
            "route": "structured",
            "query": query,
            "academic_year": academic_year,
            "semester": semester,
            "event": event,
            "reason": "Academic calendar question"
        }

    # Default to semantic retrieval
    return {
        "route": "semantic",
        "query": query,
        "academic_year": academic_year,
        "semester": semester,
        "event": None,
        "reason": "General academic knowledge question"
    }
