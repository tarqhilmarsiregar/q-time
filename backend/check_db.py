
from app.rag.database import get_connection


def main():
    try:
        with get_connection() as conn:

            database = conn.execute(
                "SELECT current_database()"
            ).fetchone()

            vector = conn.execute(
                """
                SELECT extversion
                FROM pg_extension
                WHERE extname = 'vector'
                """
            ).fetchone()

            print("Database connection: SUCCESS")
            print("Database name:", database[0])
            print("pgvector version:", vector[0] if vector else "NOT FOUND")

    except Exception:
        print("Database connection: FAILED")
        raise


if __name__ == "__main__":
    main()
