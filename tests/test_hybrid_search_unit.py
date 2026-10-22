
from unittest.mock import Mock

import pytest

from src.ingestion.chunker import CodeChunk
from src.retrieval.hybrid_search import HybridSearch


def make_chunks():
    return [
        CodeChunk(
            chunk_id="chunk_one",
            file_path="src/auth.py",
            language="python",
            content="def authenticate_user(): return True",
            start_line=1,
            end_line=1,
        ),
        CodeChunk(
            chunk_id="chunk_two",
            file_path="src/math.py",
            language="python",
            content="def add_numbers(a, b): return a + b",
            start_line=1,
            end_line=1,
        ),
    ]


def test_tokenize_lowercases_and_splits_text():
    from src.retrieval.hybrid_search import tokenize

    assert tokenize("Hello WORLD Python") == ["hello", "world", "python"]


def test_search_before_build_index_raises_error():
    search = HybridSearch(Mock())

    with pytest.raises(RuntimeError, match="build_index"):
        search.search("authenticate user")


def test_search_rejects_nonpositive_limits():
    search = HybridSearch(Mock())
    search.build_index(make_chunks())

    with pytest.raises(ValueError):
        search.search("authenticate", limit=0)

    with pytest.raises(ValueError):
        search.search("authenticate", candidate_limit=0)


def test_search_returns_ranked_results():
    vector_store = Mock()
    vector_store.search.return_value = [
        {
            "chunk_id": "chunk_one",
            "file_path": "src/auth.py",
            "language": "python",
            "content": "def authenticate_user(): return True",
            "start_line": 1,
            "end_line": 1,
            "score": 0.9,
        }
    ]

    search = HybridSearch(vector_store)
    search.build_index(make_chunks())

    results = search.search("authenticate user", limit=2)

    assert len(results) >= 1
    assert results[0]["chunk_id"] == "chunk_one"
    assert "hybrid_score" in results[0]
    vector_store.search.assert_called_once_with(
        "authenticate user",
        limit=15,
    )


def test_search_respects_result_limit():
    vector_store = Mock()
    vector_store.search.return_value = []

    search = HybridSearch(vector_store)
    search.build_index(make_chunks())

    results = search.search("python", limit=1, candidate_limit=2)

    assert len(results) <= 1
