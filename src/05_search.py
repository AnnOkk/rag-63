
import ollama
import psycopg

from pgvector import Vector
from pgvector.psycopg import register_vector


MODEL = "embeddinggemma"

DB_URL = (
    "postgresql://"
    "rag_user:rag_password"
    "@localhost:5432/"
    "rag_course"
)


def create_embedding(text: str) -> Vector:
    response = ollama.embed(
        model=MODEL,
        input=text
    )

    return Vector(
        response["embeddings"][0]
    )


def semantic_search(
    query: str,
    limit: int = 3
):
    query_embedding = create_embedding(query)

    with psycopg.connect(DB_URL) as conn:

        register_vector(conn)

        result = conn.execute(
            """
            SELECT
                title,
                content,
                embedding <=> %s
                    AS cosine_distance
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (
                query_embedding,
                query_embedding,
                limit
            )
        )

        return result.fetchall()



if __name__ == "__main__":
    queries = [
        "How can I develop a new feature without affecting the main codebase?",
        "How can I run my application with the same dependencies on different computers?",
        "How can an API reject incorrect input before processing a request?",
        "How can database transactions read data while other transactions are changing it?",
        "How can I add logging to a function without changing its original code?"
    ]

    for query in queries:
        print(f"\nQuery: {query}\n")

        results = semantic_search(query, limit=3)

        for index, (title, content, distance) in enumerate(
            results,
            start=1
        ):
            similarity = 1 - distance

            print(f"{index}. {title}")
            print(f"   Similarity: {similarity:.4f}")
            print(f"   {content}")
            print()

    