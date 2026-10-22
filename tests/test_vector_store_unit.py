
from types import SimpleNamespace

import numpy as np

from src.ingestion.chunker import CodeChunk
from src.retrieval import vector_store as vector_store_module
from src.retrieval.vector_store import VectorStore


class FakeEmbeddingModel:
    def get_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.array([1.0, 0.0, 0.0])

        return np.array([[1.0, 0.0, 0.0] for _ in texts])


class FakeQdrantClient:
    def __init__(self, **kwargs):
        self.points = []
        self.closed = False
        self.created = False

    def collection_exists(self, name):
        return False

    def create_collection(self, **kwargs):
        self.created = True

    def upsert(self, **kwargs):
        self.points.extend(kwargs["points"])

    def query_points(self, **kwargs):
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id=self.points[0].id,
                    score=0.95,
                    payload=self.points[0].payload,
                )
            ] if self.points else []
        )

    def close(self):
        self.closed = True


def make_chunk():
    return CodeChunk(
        chunk_id="12345678123456781234567812345678",
        file_path="src/app.py",
        language="python",
        content="def greet(): return 'hello'",
        start_line=1,
        end_line=1,
    )


def make_store(monkeypatch, tmp_path):
    monkeypatch.setattr(
        vector_store_module,
        "QdrantClient",
        FakeQdrantClient,
    )
    monkeypatch.setattr(
        vector_store_module,
        "SentenceTransformer",
        lambda model_name: FakeEmbeddingModel(),
    )

    return VectorStore(storage_path=str(tmp_path / "qdrant"))


def test_initializes_collection(monkeypatch, tmp_path):
    store = make_store(monkeypatch, tmp_path)

    assert store.client.created is True
    assert store.vector_size == 3

    store.close()


def test_empty_chunks_returns_zero(monkeypatch, tmp_path):
    store = make_store(monkeypatch, tmp_path)

    assert store.index_chunks([]) == 0
    assert store.client.points == []

    store.close()


def test_indexes_chunks(monkeypatch, tmp_path):
    store = make_store(monkeypatch, tmp_path)

    count = store.index_chunks([make_chunk()])

    assert count == 1
    assert len(store.client.points) == 1

    payload = store.client.points[0].payload
    assert payload["file_path"] == "src/app.py"
    assert payload["language"] == "python"
    assert payload["content"] == "def greet(): return 'hello'"
    assert payload["start_line"] == 1
    assert payload["end_line"] == 1

    store.close()


def test_search_returns_payload_and_score(monkeypatch, tmp_path):
    store = make_store(monkeypatch, tmp_path)
    store.index_chunks([make_chunk()])

    results = store.search("greet function", limit=1)

    assert len(results) == 1
    assert results[0]["file_path"] == "src/app.py"
    assert results[0]["score"] == 0.95
    assert results[0]["content"] == "def greet(): return 'hello'"

    store.close()


def test_close_closes_client(monkeypatch, tmp_path):
    store = make_store(monkeypatch, tmp_path)

    store.close()

    assert store.client.closed is True
