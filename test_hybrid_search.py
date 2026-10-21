
from pathlib import Path

from src.ingestion.chunker import chunk_documents
from src.ingestion.pipeline import ingest_repository
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridSearch


def main():
    repo_path = Path("data/repos/itsdangerous")
    documents = ingest_repository(repo_path)
    chunks = chunk_documents(documents)

    store = VectorStore()

    try:
        store.index_chunks(chunks)

        hybrid = HybridSearch(store)
        hybrid.build_index(chunks)

        queries = [
            "How does the library sign data securely?",
            "get_signature",
            "Unknown key derivation method",
        ]

        for query in queries:
            print(f"\n{'=' * 60}")
            print(f"Query: {query}")

            results = hybrid.search(query, limit=3)

            for result in results:
                print(
                    f"\nFile: {result['file_path']}"
                    f"\nLines: {result['start_line']}-"
                    f"{result['end_line']}"
                    f"\nHybrid score: "
                    f"{result['hybrid_score']:.4f}"
                )
                print(result["content"][:250])

    finally:
        store.close()


if __name__ == "__main__":
    main()
