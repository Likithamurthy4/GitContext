
import re

from rank_bm25 import BM25Okapi

from src.ingestion.chunker import CodeChunk
from src.retrieval.vector_store import VectorStore


def tokenize(text: str) -> list[str]:
    """Tokenize natural language, file paths, and code identifiers."""
    # Split camelCase identifiers before extracting tokens.
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)

    raw_tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    tokens = []

    for token in raw_tokens:
        if not token:
            continue

        # Keep the complete identifier and its individual components.
        # Example: query_vector -> query_vector, query, vector
        tokens.append(token)

        parts = [part for part in token.split("_") if part]
        if len(parts) > 1:
            tokens.extend(parts)

    return tokens


def chunk_key(item: dict) -> tuple:
    """Build a consistent identity for a chunk across search methods."""
    return (
        str(item.get("file_path", "")).replace("\\", "/"),
        item.get("start_line"),
        item.get("end_line"),
        item.get("content", ""),
    )


class HybridSearch:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.chunks: list[CodeChunk] = []
        self.bm25 = None

    def build_index(self, chunks: list[CodeChunk]) -> None:
        """Build a keyword index over code, identifiers, and file paths."""
        self.chunks = chunks

        if not chunks:
            self.bm25 = None
            return

        tokenized_chunks = [
            tokenize(
                f"{chunk.file_path} "
                f"{chunk.language} "
                f"{chunk.content}"
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
        """Combine semantic and keyword results using Reciprocal Rank Fusion."""
        if self.bm25 is None:
            raise RuntimeError("Call build_index() before searching.")

        if limit < 1 or candidate_limit < 1:
            raise ValueError("Search limits must be positive.")

        # Retrieve semantic candidates.
        semantic_results = self.vector_store.search(
            query,
            limit=candidate_limit,
        )

        # Retrieve keyword candidates.
        keyword_scores = self.bm25.get_scores(tokenize(query))

        keyword_indices = sorted(
            range(len(keyword_scores)),
            key=lambda index: keyword_scores[index],
            reverse=True,
        )[:candidate_limit]

        # Merge both rankings using a stable chunk identity.
        # Qdrant UUIDs and original chunk IDs can differ, so do not
        # rely on chunk_id alone to identify the same indexed chunk.
        rank_constant = 60
        fused: dict[tuple, dict] = {}

        for rank, result in enumerate(semantic_results, start=1):
            key = chunk_key(result)

            if key not in fused:
                fused[key] = {
                    "result": result,
                    "score": 0.0,
                }

            fused[key]["score"] += 1.0 / (rank_constant + rank)

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

            key = chunk_key(result)

            if key not in fused:
                fused[key] = {
                    "result": result,
                    "score": 0.0,
                }

            fused[key]["score"] += 1.0 / (rank_constant + rank)

        ranked = sorted(
            fused.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        return [
            {
                **item["result"],
                "hybrid_score": item["score"],
            }
            for item in ranked[:limit]
        ]
