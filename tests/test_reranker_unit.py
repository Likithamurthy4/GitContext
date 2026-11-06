
import numpy as np

from src.retrieval import reranker as reranker_module
from src.retrieval.reranker import CodeReranker


class FakeCrossEncoder:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = FakeTokenizer()

    def predict(self, pairs):
        # Higher score for the first candidate.
        return np.array([0.9, 0.2][:len(pairs)])


class FakeTokenizer:
    model_max_length = 512

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def num_special_tokens_to_add(self, pair):
        return 2

    def decode(
        self,
        tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        return " ".join(tokens)


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


def test_reranker_scores_late_content_in_long_chunks(monkeypatch):
    class TailAwareCrossEncoder(FakeCrossEncoder):
        def __init__(self, model_name):
            super().__init__(model_name)
            self.scored_pairs = []

        def predict(self, pairs):
            self.scored_pairs.extend(pairs)
            return np.array(
                [
                    0.95 if "TAIL_MARKER" in candidate else 0.1
                    for _, candidate in pairs
                ]
            )

    monkeypatch.setattr(
        reranker_module,
        "CrossEncoder",
        TailAwareCrossEncoder,
    )
    reranker = CodeReranker()
    candidates = [
        {
            "chunk_id": "long",
            "file_path": "src/long.py",
            "language": "python",
            "content": " ".join(["early"] * 30 + ["TAIL_MARKER"]),
            "start_line": 1,
            "end_line": 80,
        },
        {
            "chunk_id": "short",
            "file_path": "src/short.py",
            "language": "python",
            "content": "unrelated",
            "start_line": 1,
            "end_line": 1,
        },
    ]

    results = reranker.rerank("find late marker", candidates, limit=2)

    long_result = next(item for item in results if item["chunk_id"] == "long")
    short_result = next(item for item in results if item["chunk_id"] == "short")
    assert long_result["rerank_score"] == 0.95
    assert short_result["rerank_score"] == 0.1
    assert any("TAIL_MARKER" in candidate for _, candidate in reranker.model.scored_pairs)
