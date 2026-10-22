
from fastapi.testclient import TestClient

from src import main


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
        def generate(self, question, chunks):
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


def test_ask_converts_gemini_quota_error_to_429(monkeypatch):
    class FakeHybrid:
        def search(self, question, limit, candidate_limit):
            return []

    class FakeReranker:
        def rerank(self, question, candidates, limit):
            return []

    class FakeGenerator:
        def generate(self, question, chunks):
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
        def generate(self, question, chunks):
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
