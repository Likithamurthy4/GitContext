
from pathlib import Path

from src.ingestion.chunker import chunk_documents
from src.ingestion.pipeline import ingest_repository
from src.retrieval.vector_store import VectorStore


def main():
    repo_path = Path("data/repos/itsdangerous")

    documents = ingest_repository(repo_path)
    chunks = chunk_documents(documents)

    store = VectorStore()
    try:
        count = store.index_chunks(chunks)
        print(f"Chunks indexed: {count}")

        results = store.search(
            "How does the library sign data securely?",
            limit=5,
        )

        print("\nSearch results:")
        for result in results:
            print(
                f"\nFile: {result['file_path']}"
                f"\nLines: {result['start_line']}-"
                f"{result['end_line']}"
                f"\nScore: {result['score']:.4f}"
            )
            print(result["content"][:300])
    finally:
        store.close()


if __name__ == "__main__":
    main()
