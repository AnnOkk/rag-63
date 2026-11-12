
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


documents = [
    {
        "title": "Python asyncio",
        "content": (
            "Python's asyncio library coordinates coroutines "
            "using an event loop. Tasks can yield control while "
            "waiting for I/O, allowing other tasks to run."
        )
    },
    {
        "title": "JavaScript Event Loop",
        "content": (
            "JavaScript uses an event loop to coordinate "
            "asynchronous operations and execute queued callbacks."
        )
    },
    {
        "title": "FastAPI",
        "content": (
            "FastAPI is a Python framework for building web APIs. "
            "It supports asynchronous request handlers using "
            "async and await."
        )
    },
    {
        "title": "PostgreSQL Transactions",
        "content": (
            "PostgreSQL transactions group multiple database "
            "operations into a single atomic unit of work."
        )
    },
    {
        "title": "PostgreSQL Indexes",
        "content": (
            "Database indexes can improve query performance by "
            "allowing PostgreSQL to find rows without scanning "
            "the entire table."
        )
    },
    {
        "title": "MongoDB",
        "content": (
            "MongoDB stores information as flexible "
            "document-like records rather than rows in "
            "traditional relational tables."
        )
    },
    {
        "title": "React",
        "content": (
            "React applications are built from components. "
            "Components can manage state and render user "
            "interfaces when data changes."
        )
    },
    {
        "title": "Redux Toolkit",
        "content": (
            "Redux Toolkit provides utilities for managing "
            "shared application state in JavaScript applications."
        )
    },
    {
        "title": "Docker",
        "content": (
            "Docker packages applications and their dependencies "
            "into containers that can run consistently across "
            "different environments."
        )
    },
    {
        "title": "REST API",
        "content": (
            "REST APIs expose resources through HTTP endpoints "
            "and commonly use methods such as GET, POST, PUT, "
            "PATCH, and DELETE."
        )
    },
    {
        "title": "Git Branches",
        "content": (
            "Git branches allow developers to work on separate "
            "features without changing the main development line. "
            "Changes can later be merged into another branch."
        )
    },
    {
        "title": "Docker Containers",
        "content": (
            "Docker containers package applications with their "
            "dependencies, allowing them to run consistently "
            "across different environments."
        )
    },
    {
        "title": "FastAPI Validation",
        "content": (
            "FastAPI uses Pydantic models to validate incoming "
            "request data. Invalid data produces a validation "
            "error before the endpoint logic executes."
        )
    },
    {
        "title": "PostgreSQL MVCC",
        "content": (
            "PostgreSQL uses Multi-Version Concurrency Control "
            "to allow transactions to read consistent snapshots "
            "of data while other transactions modify records."
        )
    },
    {
        "title": "Python Decorators",
        "content": (
            "Python decorators wrap functions to extend their "
            "behavior without modifying the original function. "
            "They are commonly used for logging and authentication."
        )
    },
    {
        "title": "Python Generators",
        "content": (
            "Python generators produce values one at a time using "
            "the yield keyword. They allow programs to process "
            "large sequences without storing everything in memory."
        )
    },
    {
        "title": "Python Context Managers",
        "content": (
            "Python context managers manage resources such as "
            "files and database connections. The with statement ensures "
            "that cleanup happens when a block finishes."
        )
    },
    {
        "title": "Python Exceptions",
        "content": (
            "Python exceptions allow programs to handle errors "
            "using try and except blocks instead of terminating "
            "unexpectedly."
        )
    },
    {
        "title": "FastAPI Dependency Injection",
        "content": (
            "FastAPI dependency injection allows endpoints to "
            "receive reusable components such as database sessions "
            "and authentication services through Depends."
        )
    },
    {
        "title": "FastAPI Middleware",
        "content": (
            "FastAPI middleware processes HTTP requests before "
            "they reach endpoints and responses before they "
            "are returned to clients."
        )
    },
    {
        "title": "HTTP PATCH",
        "content": (
            "The HTTP PATCH method applies partial modifications "
            "to an existing resource without replacing the entire "
            "resource representation."
        )
    },
    {
        "title": "PostgreSQL Isolation Levels",
        "content": (
            "PostgreSQL isolation levels control how concurrent "
            "transactions observe database changes. Read Committed, "
            "Repeatable Read, and Serializable provide different "
            "levels of consistency."
        )
    },
    {
        "title": "PostgreSQL WAL",
        "content": (
            "PostgreSQL Write-Ahead Logging records database changes "
            "before modified data pages are written to disk. "
            "This supports crash recovery and durability."
        )
    },
    {
        "title": "PostgreSQL VACUUM",
        "content": (
            "PostgreSQL VACUUM removes dead row versions created "
            "by updates and deletes, allowing storage space "
            "to be reused and helping maintain performance."
        )
    },
    {
        "title": "Git Merge Conflicts",
        "content": (
            "Git merge conflicts occur when changes from different "
            "branches cannot be combined automatically. Developers "
            "must resolve conflicting changes before completing "
            "the merge."
        )
    },
    {
        "title": "Git Pull Requests",
        "content": (
            "Pull requests allow developers to propose code changes "
            "for review before merging them into a shared branch."
        )
    },
    {
        "title": "Docker Volumes",
        "content": (
            "Docker volumes store persistent data independently "
            "of containers, allowing information to survive "
            "container removal and recreation."
        )
    },
    {
        "title": "Docker Networks",
        "content": (
            "Docker networks enable containers to communicate "
            "with each other while providing network isolation."
        )
    },
    {
        "title": "React Hooks",
        "content": (
            "React Hooks allow function components to manage state "
            "and side effects. Common hooks include useState "
            "and useEffect."
        )
    },
    {
        "title": "RTK Query",
        "content": (
            "RTK Query simplifies fetching and caching server data "
            "in Redux applications. It manages loading states, "
            "cached responses, and automatic refetching."
        )
    }
]


def index_documents():
    new_documents = documents

    texts = [
        document["content"]
        for document in new_documents
    ]

    response = ollama.embed(
        model=MODEL,
        input=texts
    )

    embeddings = response["embeddings"]

    if len(embeddings) != len(new_documents):
        raise ValueError(
            "Number of embeddings does not match documents"
        )

    with psycopg.connect(
        DB_URL,
        autocommit=True
    ) as conn:

        conn.execute(
            "CREATE EXTENSION IF NOT EXISTS vector"
        )

        register_vector(conn)

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id BIGSERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding VECTOR(768) NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
                documents_title_unique_idx
            ON documents (title)
            """
        )

        for document, embedding in zip(
            new_documents,
            embeddings
        ):
            conn.execute(
                """
                INSERT INTO documents (
                    title,
                    content,
                    embedding
                )
                VALUES (%s, %s, %s)
                ON CONFLICT (title) DO NOTHING
                """,
                (
                    document["title"],
                    document["content"],
                    Vector(embedding)
                )
            )

    print(f"Processed {len(new_documents)} documents.")


if __name__ == "__main__":
    index_documents()
