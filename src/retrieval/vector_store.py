from uuid import UUID
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from src.ingestion.chunker import CodeChunk


class VectorStore:
    def __init__(
        self,
        storage_path: str = "data/qdrant",
        collection_name: str = "gitcontext_code",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.client = QdrantClient(path=storage_path)
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name)
        self.vector_size = self.model.get_embedding_dimension()

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def index_chunks(self, chunks: list[CodeChunk]) -> int:
        if not chunks:
            return 0

        texts = [
            f"File: {chunk.file_path}\n"
            f"Language: {chunk.language}\n"
            f"Code:\n{chunk.content}"
            for chunk in chunks
        ]

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        points = []

        for chunk, vector in zip(chunks, vectors):
            points.append(
                models.PointStruct(
                    id=str(
                        __import__("uuid").UUID(
                            chunk.chunk_id[:32]
                        )
                    ),
                    vector=vector.tolist(),
                    payload={
                        "file_path": chunk.file_path,
                        "language": chunk.language,
                        "content": chunk.content,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        return len(points)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        query_vector = self.model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        ).points

        return [
            {
                "chunk_id": str(result.id),
                "score": result.score,
                **result.payload,
            }
            for result in results
        ]

    def close(self):
        self.client.close()
