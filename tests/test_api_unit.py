
from fastapi.testclient import TestClient
import pytest

from src import main


def test_project_overview_retrieval_covers_distinct_facets():
    class FakeHybrid:
        def __init__(self):
            self.queries = []

        def search(self, query, limit, candidate_limit):
            self.queries.append((query, limit, candidate_limit))
            facet_index = len(self.queries)
            shared = {
                "file_path": "README.md",
                "start_line": 1,
                "end_line": 5,
                "content": "Project overview",
            }
            facet = {
                "file_path": f"docs/facet-{facet_index}.md",
                "start_line": 1,
                "end_line": 5,
                "content": f"Facet {facet_index}",
            }
            return [shared, facet]

    class FakeReranker:
        def __init__(self):
            self.candidates = []
            self.limit = None

        def rerank(self, query, candidates, limit):
            self.candidates = candidates
            self.limit = limit
            return candidates[:limit]

    hybrid = FakeHybrid()
    reranker = FakeReranker()

    result = main.retrieve_relevant_chunks(
        hybrid,
        reranker,
        "explain whole project detailly",
        "explain whole project detailly",
    )

    assert len(hybrid.queries) == 5
    assert all(
        limit == 8 and candidate_limit == 20
        for _, limit, candidate_limit in hybrid.queries
    )
    assert len(reranker.candidates) == 6
    assert reranker.limit == 16
    assert len(result) == 6


@pytest.fixture(autouse=True)
def clear_conversation_history():
    main.app.state.conversation_histories = {}


def test_root_endpoint():
    client = TestClient(main.app)

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to GitContext API",
        "docs": "/docs",
    }


def test_health_endpoint():
    client = TestClient(main.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_rejects_empty_question():
    client = TestClient(main.app)

    response = client.post(
        "/ask",
        json={"question": ""},
    )

    assert response.status_code == 422


def test_ask_rejects_question_that_is_too_short():
    client = TestClient(main.app)

    response = client.post(
        "/ask",
        json={"question": "hi"},
    )

    assert response.status_code == 422
    assert "longer question" in response.json()["detail"]


def test_ask_rejects_invalid_repository_url():
    client = TestClient(main.app)

    response = client.post(
        "/ask",
        json={
            "question": "How does authentication work?",
            "repo_url": "https://example.com/not-github",
        },
    )

    assert response.status_code == 400
    assert "valid public GitHub repository URL" in response.json()["detail"]


@pytest.mark.parametrize(
    "repo_url",
    [
        "http://github.com/example/repo",
        "https://github.com/example/repo/issues",
        "https://github.com/example/repo?tab=readme",
        "https://attacker@github.com/example/repo",
    ],
)
def test_ask_rejects_malformed_github_urls(repo_url):
    client = TestClient(main.app)

    response = client.post(
        "/ask",
        json={
            "question": "How does authentication work?",
            "repo_url": repo_url,
        },
    )

    assert response.status_code == 400
    assert "valid public GitHub repository URL" in response.json()["detail"]


def test_ask_returns_answer_with_mocked_pipeline(monkeypatch):
    class FakeHybrid:
        def search(self, question, limit, candidate_limit):
            return [
                {
                    "chunk_id": "chunk1",
                    "file_path": "src/auth.py",
                    "language": "python",
                    "content": "def authenticate(): pass",
                    "start_line": 1,
                    "end_line": 1,
                }
            ]

    class FakeReranker:
        def rerank(self, question, candidates, limit):
            return candidates[:limit]

    class FakeGenerator:
        def rewrite_question(self, question, conversation_history=None):
            return question

        def generate(
            self,
            question,
            chunks,
            conversation_history=None,
        ):
            return {
                "answer": "Authentication is handled in src/auth.py.",
                "sources": [
                    {
                        "file_path": "src/auth.py",
                        "start_line": 1,
                        "end_line": 1,
                    }
                ],
            }

    monkeypatch.setattr(
        main,
        "build_repository_context",
        lambda repo_url, app: {"hybrid": FakeHybrid()},
    )

    main.app.state.reranker = FakeReranker()
    main.app.state.generator = FakeGenerator()

    client = TestClient(main.app)

    response = client.post(
        "/ask",
        json={
            "question": "How does authentication work?",
            "repo_url": "https://github.com/example/repo",
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["question"] == "How does authentication work?"
    assert body["repository"] == "https://github.com/example/repo"
    assert "Authentication is handled" in body["answer"]
    assert body["sources"][0]["file_path"] == "src/auth.py"


def test_ask_reuses_conversation_for_follow_up_questions(monkeypatch):
    class FakeHybrid:
        def __init__(self):
            self.queries = []
            self.search_limits = []

        def search(self, question, limit, candidate_limit):
            self.queries.append(question)
            self.search_limits.append((limit, candidate_limit))
            return [
                {
                    "chunk_id": "chunk1",
                    "file_path": "src/auth.py",
                    "language": "python",
                    "content": "def authenticate(token): return token",
                    "start_line": 1,
                    "end_line": 1,
                }
            ]

    class FakeReranker:
        def __init__(self):
            self.limits = []

        def rerank(self, question, candidates, limit):
            self.limits.append(limit)
            return candidates[:limit]

    class FakeGenerator:
        def __init__(self):
            self.rewrite_histories = []
            self.answer_histories = []

        def rewrite_question(self, question, conversation_history=None):
            history = list(conversation_history or [])
            self.rewrite_histories.append(history)
            return (
                "How does authenticate validate the token?"
                if history
                else question
            )

        def generate(
            self,
            question,
            chunks,
            conversation_history=None,
        ):
            self.answer_histories.append(
                list(conversation_history or [])
            )
            return {
                "answer": f"Answer to: {question}",
                "sources": [
                    {
                        "source_id": "S1",
                        "file_path": "src/auth.py",
                        "start_line": 1,
                        "end_line": 1,
                    }
                ],
            }

    hybrid = FakeHybrid()
    generator = FakeGenerator()
    reranker = FakeReranker()
    monkeypatch.setattr(
        main,
        "build_repository_context",
        lambda repo_url, app: {"hybrid": hybrid},
    )
    main.app.state.reranker = reranker
    main.app.state.generator = generator
    main.app.state.conversation_histories = {}
    client = TestClient(main.app)
    conversation_id = "follow-up-conversation"
    repository = "https://github.com/example/repo"

    first = client.post(
        "/ask",
        json={
            "question": "How does authentication work?",
            "repo_url": repository,
            "conversation_id": conversation_id,
        },
    )
    second = client.post(
        "/ask",
        json={
            "question": "What does it validate?",
            "repo_url": repository,
            "conversation_id": conversation_id,
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["conversation_id"] == conversation_id
    assert second.json()["conversation_id"] == conversation_id
    assert hybrid.queries == [
        "How does authentication work?",
        "How does authenticate validate the token?",
    ]
    assert hybrid.search_limits == [(12, 20), (12, 20)]
    assert reranker.limits == [8, 8]
    assert generator.rewrite_histories[0] == []
    assert generator.rewrite_histories[1] == [
        {
            "role": "user",
            "content": "How does authentication work?",
        },
        {
            "role": "assistant",
            "content": "Answer to: How does authentication work?",
        },
    ]
    assert generator.answer_histories[1] == generator.rewrite_histories[1]


def test_ask_converts_gemini_quota_error_to_429(monkeypatch):
    class FakeHybrid:
        def search(self, question, limit, candidate_limit):
            return []

    class FakeReranker:
        def rerank(self, question, candidates, limit):
            return []

    class FakeGenerator:
        def rewrite_question(self, question, conversation_history=None):
            return question

        def generate(
            self,
            question,
            chunks,
            conversation_history=None,
        ):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(
        main,
        "build_repository_context",
        lambda repo_url, app: {"hybrid": FakeHybrid()},
    )

    main.app.state.reranker = FakeReranker()
    main.app.state.generator = FakeGenerator()

    client = TestClient(main.app)

    response = client.post(
        "/ask",
        json={
            "question": "Explain the project structure",
            "repo_url": "https://github.com/example/repo",
        },
    )

    assert response.status_code == 429
    assert "quota exceeded" in response.json()["detail"]

def test_ask_converts_gemini_unavailable_error_to_503(monkeypatch):
    class FakeHybrid:
        def search(self, question, limit, candidate_limit):
            return []

    class FakeReranker:
        def rerank(self, question, candidates, limit):
            return []

    class FakeGenerator:
        def rewrite_question(self, question, conversation_history=None):
            return question

        def generate(
            self,
            question,
            chunks,
            conversation_history=None,
        ):
            raise RuntimeError(
                "503 UNAVAILABLE: model is experiencing high demand"
            )

    monkeypatch.setattr(
        main,
        "build_repository_context",
        lambda repo_url, app: {"hybrid": FakeHybrid()},
    )

    main.app.state.reranker = FakeReranker()
    main.app.state.generator = FakeGenerator()

    client = TestClient(main.app)

    response = client.post(
        "/ask",
        json={
            "question": "Explain the project structure",
            "repo_url": "https://github.com/example/repo",
        },
    )

    assert response.status_code == 503
    assert (
        "temporarily unavailable"
        in response.json()["detail"]
    )
