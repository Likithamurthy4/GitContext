
from pathlib import Path

from src.ingestion.chunker import chunk_documents
from src.ingestion.pipeline import ingest_repository
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import CodeReranker
from src.llm.answer_generator import AnswerGenerator


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
        ranked_chunks = reranker.rerank(
            query,
            candidates,
            limit=5,
        )

        generator = AnswerGenerator()
        result = generator.generate(query, ranked_chunks)

        print("\nQUESTION:")
        print(query)

        print("\nANSWER:")
        print(result["answer"])

        print("\nSOURCES:")
        for source in result["sources"]:
            print(
                f"[{source['source_id']}] "
                f"{source['file_path']}:"
                f"{source['start_line']}-"
                f"{source['end_line']}"
            )

    finally:
        store.close()


if __name__ == "__main__":
    main()
