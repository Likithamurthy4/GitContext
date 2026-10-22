
import hashlib
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.ingestion.repository_loader import clone_repository
from src.ingestion.pipeline import ingest_repository
from src.ingestion.chunker import chunk_documents
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import CodeReranker
from src.llm.answer_generator import AnswerGenerator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPOSITORIES_DIR = PROJECT_ROOT / "data" / "repos"
VECTOR_DB_DIR = PROJECT_ROOT / "data" / "qdrant"

DEFAULT_REPO_URL = "https://github.com/pallets/itsdangerous"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    repo_url: str = Field(default=DEFAULT_REPO_URL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Loading reranker...")
    app.state.reranker = CodeReranker()

    print("[STARTUP] Initializing answer generator...")
    app.state.generator = AnswerGenerator()

    # Cache repository search contexts for this server process.
    app.state.repository_contexts = {}

    print("[STARTUP] GitContext is ready.")
    yield

    # Close persistent vector-store clients on shutdown.
    for context in app.state.repository_contexts.values():
        try:
            context["vector_store"].close()
        except Exception as exc:
            print(f"[SHUTDOWN] Could not close vector store: {exc}")


app = FastAPI(
    title="GitContext API",
    description="Ask questions about a GitHub codebase.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_repository_context(repo_url: str, app: FastAPI) -> dict:
    repo_url = repo_url.strip().rstrip("/")

    if not re.fullmatch(
        r"https://github\.com/[\w.-]+/[\w.-]+(?:\.git)?",
        repo_url,
    ):
        raise ValueError(
            "Enter a valid public GitHub repository URL."
        )

    # Normalize .git so equivalent URLs share a cache key.
    normalized_url = repo_url.removesuffix(".git")
    repo_key = hashlib.sha256(
        normalized_url.encode("utf-8")
    ).hexdigest()[:16]

    cached = app.state.repository_contexts.get(repo_key)

    if cached is not None:
        print("[CACHE] Reusing indexed repository.")
        return cached

    print("[INGEST] Preparing repository:", repo_url)
    repo_name = normalized_url.rsplit("/", 1)[-1]
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", repo_name)
    repo_path = REPOSITORIES_DIR / f"{safe_name}_{repo_key}"

    REPOSITORIES_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    if not (repo_path / ".git").exists():
        if repo_path.exists():
            raise ValueError(
                "The repository destination exists but is not "
                "a Git clone. Please inspect this folder: "
                f"{repo_path}"
            )

        print("[INGEST] Cloning repository...")
        clone_repository(normalized_url, repo_path)

    print("[INGEST] Scanning repository files...")
    documents = ingest_repository(repo_path)

    if not documents:
        raise ValueError(
            "No supported source files were found in this repository."
        )

    print(f"[INGEST] Found {len(documents)} files.")
    chunks = chunk_documents(documents, strategy="ast")

    if not chunks:
        raise ValueError(
            "The repository did not produce searchable code chunks."
        )

    print(f"[INGEST] Created {len(chunks)} chunks.")

    vector_store = VectorStore(
        storage_path=str(VECTOR_DB_DIR / repo_key),
        collection_name=f"gitcontext_{repo_key}",
    )

    try:
        print("[INGEST] Creating embeddings and indexing chunks...")
        indexed_count = vector_store.index_chunks(chunks)
        print(f"[INGEST] Indexed {indexed_count} chunks.")

        hybrid_search = HybridSearch(vector_store)
        hybrid_search.build_index(chunks)

        context = {
            "hybrid": hybrid_search,
            "vector_store": vector_store,
        }

        app.state.repository_contexts[repo_key] = context
        print("[INGEST] Repository indexing complete.")

        return context

    except Exception:
        vector_store.close()
        raise


@app.get("/")
def root():
    return {
        "message": "Welcome to GitContext API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask_question(request: AskRequest):
    question = request.question.strip()

    if len(question) < 3:
        raise HTTPException(
            status_code=422,
            detail="Please enter a longer question.",
        )

    start = time.perf_counter()

    try:
        print("[TIMING] Starting repository context.")

        context = build_repository_context(
            request.repo_url,
            app,
        )

        print(
            f"[TIMING] Repository setup: "
            f"{time.perf_counter() - start:.2f}s"
        )

        stage = time.perf_counter()
        candidates = context["hybrid"].search(
            question,
            limit=5,
            candidate_limit=8,
        )

        print(
            f"[TIMING] Search: "
            f"{time.perf_counter() - stage:.2f}s"
        )

        stage = time.perf_counter()
        ranked_chunks = app.state.reranker.rerank(
            question,
            candidates,
            limit=3,
        )

        print(
            f"[TIMING] Reranking: "
            f"{time.perf_counter() - stage:.2f}s"
        )

        stage = time.perf_counter()
        result = app.state.generator.generate(
            question,
            ranked_chunks,
        )

        print(
            f"[TIMING] Answer generation: "
            f"{time.perf_counter() - stage:.2f}s"
        )

        print(
            f"[TIMING] Total request: "
            f"{time.perf_counter() - start:.2f}s"
        )

        return {
            "question": question,
            "repository": request.repo_url,
            "answer": result["answer"],
            "sources": result["sources"],
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        error_message = str(exc)
        print(f"[ERROR] Question answering failed: {error_message}")

        if (
            "429" in error_message
            or "RESOURCE_EXHAUSTED" in error_message
        ):
            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini API quota exceeded. Please check your "
                    "API usage limits and try again when quota is available."
                ),
            ) from exc

        if (
            "503" in error_message
            or "UNAVAILABLE" in error_message
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    "The answer-generation model is temporarily unavailable. "
                    "Please try again shortly."
                ),
            ) from exc

        raise HTTPException(
            status_code=500,
            detail="Unable to answer the question. Check server logs.",
        ) from exc
