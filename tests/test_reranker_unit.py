
import numpy as np

from src.retrieval import reranker as reranker_module
from src.retrieval.reranker import CodeReranker


class FakeCrossEncoder:
    def __init__(self, model_name):
        self.model_name = model_name

    def predict(self, pairs):
        # Higher score for the first candidate.
        return np.array([0.9, 0.2][:len(pairs)])


def make_candidates():
    return [
        {
            "chunk_id": "one",
            "file_path": "src/auth.py",
            "language": "python",
            "content": "authenticate user",
            "start_line": 1,
            "end_line": 2,
        },
        {
            "chunk_id": "two",
            "file_path": "src/math.py",
            "language": "python",
            "content": "add numbers",
            "start_line": 1,
            "end_line": 2,
        },
    ]


def make_reranker(monkeypatch):
    monkeypatch.setattr(
        reranker_module,
        "CrossEncoder",
        FakeCrossEncoder,
    )
    return CodeReranker()


def test_reranks_candidates_by_score(monkeypatch):
    reranker = make_reranker(monkeypatch)

    results = reranker.rerank(
        "authenticate a user",
        make_candidates(),
        limit=2,
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "one"
    assert results[0]["rerank_score"] == 0.9
    assert results[1]["rerank_score"] == 0.2


def test_empty_candidates_returns_empty_list(monkeypatch):
    reranker = make_reranker(monkeypatch)

    assert reranker.rerank("question", [], limit=3) == []


def test_nonpositive_limit_returns_empty_list(monkeypatch):
    reranker = make_reranker(monkeypatch)

    assert reranker.rerank(
        "question",
        make_candidates(),
        limit=0,
    ) == []


def test_reranker_respects_limit(monkeypatch):
    reranker = make_reranker(monkeypatch)

    results = reranker.rerank(
        "question",
        make_candidates(),
        limit=1,
    )

    assert len(results) == 1
