
import psycopg

from pgvector.psycopg import register_vector

from app.rag.config import DATABASE_URL


def get_connection():
    """
    Create a PostgreSQL connection
    and register pgvector adapter.
    """

    connection = psycopg.connect(DATABASE_URL)

    register_vector(connection)

    return connection
