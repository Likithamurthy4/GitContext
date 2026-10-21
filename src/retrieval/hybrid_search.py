
from rank_bm25 import BM25Okapi

from src.retrieval.vector_store import VectorStore


def tokenize(text: str) -> list[str]:
    return text.lower().split()


class HybridSearch:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.bm25 = None
        self.chunks = []

    def build_index(self, chunks):
        self.chunks = chunks
        tokenized_chunks = [
            tokenize(
                f"{chunk.file_path} {chunk.language} {chunk.content}"
            )
            for chunk in chunks
        ]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def search(
        self,
        query: str,
        limit: int = 5,
        candidate_limit: int = 15,
    ) -> list[dict]:
        if self.bm25 is None:
            raise RuntimeError("Call build_index() before searching.")

        if limit < 1 or candidate_limit < 1:
            raise ValueError("Search limits must be positive.")

        # Semantic candidates from Qdrant.
        semantic_results = self.vector_store.search(
            query, limit=candidate_limit
        )

        # Keyword candidates from BM25.
        scores = self.bm25.get_scores(tokenize(query))
        keyword_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:candidate_limit]

        # Reciprocal Rank Fusion combines the rankings.
        fused = {}

        for rank, result in enumerate(semantic_results, start=1):
            key = result["chunk_id"]
            fused.setdefault(key, {"result": result, "score": 0.0})
            fused[key]["score"] += 1 / (60 + rank)

        for rank, index in enumerate(keyword_indices, start=1):
            chunk = self.chunks[index]
            result = {
                "chunk_id": chunk.chunk_id,
                "file_path": chunk.file_path,
                "language": chunk.language,
                "content": chunk.content,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            }
            key = chunk.chunk_id
            fused.setdefault(key, {"result": result, "score": 0.0})
            fused[key]["score"] += 1 / (60 + rank)

        ranked = sorted(
            fused.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        return [
            {**item["result"], "hybrid_score": item["score"]}
            for item in ranked[:limit]
        ]
