
from pathlib import Path

from src.ingestion.chunker import chunk_documents
from src.ingestion.pipeline import ingest_repository
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import CodeReranker


def main():
    repo_path = Path("data/repos/itsdangerous")
    documents = ingest_repository(repo_path)
    chunks = chunk_documents(documents)

    store = VectorStore()

    try:
        store.index_chunks(chunks)

        hybrid = HybridSearch(store)
        hybrid.build_index(chunks)

        query = "How does the library generate a signature?"
        candidates = hybrid.search(
            query,
            limit=10,
            candidate_limit=20,
        )

        reranker = CodeReranker()
        results = reranker.rerank(query, candidates, limit=5)

        print(f"\nQuery: {query}")
        print("\nReranked results:")

        for result in results:
            print(f"\nFile: {result['file_path']}")
            print(
                f"Lines: {result['start_line']}-"
                f"{result['end_line']}"
            )
            print(f"Rerank score: {result['rerank_score']:.4f}")
            print(result["content"][:250])

    finally:
        store.close()


if __name__ == "__main__":
    main()
