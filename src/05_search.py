
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
    while True:
        query = input("\nSearch (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Please enter a search query.")
            continue

        try:
            limit = int(input("Top K (default 3): ") or 3)
            if limit < 1:
                print("Top K must be positive.")
                continue
        except ValueError:
            print("Please enter a valid number.")
            continue

        results = semantic_search(query, limit=limit)

        for index, (title, content, distance) in enumerate(
            results, start=1
        ):
            similarity = 1 - distance

            print(f"\n{index}. {title}")
            print(f"   Similarity: {similarity:.4f}")
            print(f"   {content}")

    