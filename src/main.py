import uuid
import random
import hashlib
import os
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
from src.retrieval.hybrid_search import HybridSearch, chunk_key
from src.retrieval.reranker import CodeReranker
from src.llm.answer_generator import (
    AnswerGenerator,
    is_project_overview_question,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPOSITORIES_DIR = PROJECT_ROOT / "data" / "repos"
VECTOR_DB_DIR = PROJECT_ROOT / "data" / "qdrant"

DEFAULT_REPO_URL = "https://github.com/pallets/itsdangerous"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    repo_url: str = Field(default=DEFAULT_REPO_URL)
    conversation_id: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Loading reranker...")
    app.state.reranker = CodeReranker()

    print("[STARTUP] Initializing answer generator...")
    app.state.generator = AnswerGenerator()

    app.state.repository_contexts = {}
    app.state.conversation_histories = {}

    print("[STARTUP] GitContext is ready.")

    yield

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
        origin.strip().rstrip("/")
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
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

    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        repo_name,
    )

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


def generate_answer_with_retry(
    generator: AnswerGenerator,
    question: str,
    ranked_chunks: list[dict],
    conversation_history: list[dict] | None = None,
) -> dict:
    """Retry once when Gemini is temporarily unavailable."""

    for attempt in range(2):
        try:
            return generator.generate(
                question,
                ranked_chunks,
                conversation_history=conversation_history,
            )

        except Exception as exc:
            error_message = str(exc).upper()

            is_temporary_unavailable = (
                "503" in error_message
                or "UNAVAILABLE" in error_message
            )

            if not is_temporary_unavailable or attempt == 1:
                raise

            delay = 2 + random.uniform(0, 1)

            print(
                f"[RETRY] Gemini temporarily unavailable. "
                f"Retrying once in {delay:.1f}s."
            )

            time.sleep(delay)

    raise RuntimeError("Answer generation failed unexpectedly.")


def retrieve_relevant_chunks(
    hybrid: HybridSearch,
    reranker: CodeReranker,
    question: str,
    search_question: str,
) -> list[dict]:
    """Use focused retrieval facets for project-wide overview questions."""

    normalized_question = question.replace("\\", "/").casefold()
    indexed_chunks = getattr(hybrid, "chunks", [])
    indexed_paths = {
        str(chunk.file_path).replace("\\", "/").casefold()
        for chunk in indexed_chunks
    }
    matched_paths = {
        path for path in indexed_paths if path in normalized_question
    }

    if not matched_paths:
        basename_matches: dict[str, set[str]] = {}
        for path in indexed_paths:
            basename = path.rsplit("/", 1)[-1]
            if basename and re.search(
                rf"(?<![\w.-]){re.escape(basename)}(?![\w.-])",
                normalized_question,
            ):
                basename_matches.setdefault(basename, set()).add(path)

        matched_paths = {
            next(iter(paths))
            for paths in basename_matches.values()
            if len(paths) == 1
        }

    if matched_paths:
        file_chunks = [
            {
                "chunk_id": chunk.chunk_id,
                "file_path": chunk.file_path,
                "language": chunk.language,
                "content": chunk.content,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            }
            for chunk in indexed_chunks
            if str(chunk.file_path).replace("\\", "/").casefold()
            in matched_paths
        ]
        print(
            "[RETRIEVAL] Exact file match: "
            f"{len(matched_paths)} file(s), {len(file_chunks)} chunk(s)."
        )
        return reranker.rerank(
            search_question,
            file_chunks,
            limit=len(file_chunks),
        )

    if not is_project_overview_question(question):
        candidates = hybrid.search(
            search_question,
            limit=12,
            candidate_limit=20,
        )
        return reranker.rerank(
            search_question,
            candidates,
            limit=8,
        )

    facets = (
        "project purpose architecture components entry point",
        "end-to-end workflow pipeline execution stages",
        "important directories modules scripts responsibilities",
        "data sources inputs processing analytics outputs reports",
        "technology dependencies configuration setup run tests deployment",
    )
    candidates_by_key = {}

    for facet in facets:
        for candidate in hybrid.search(
            f"{search_question} {facet}",
            limit=8,
            candidate_limit=20,
        ):
            candidates_by_key.setdefault(
                chunk_key(candidate),
                candidate,
            )

    return reranker.rerank(
        search_question,
        list(candidates_by_key.values()),
        limit=16,
    )


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

        conversation_id = (
            request.conversation_id or str(uuid.uuid4())
        )

        normalized_repo_url = (
            request.repo_url.strip().rstrip("/")
        ).removesuffix(".git")

        conversation = app.state.conversation_histories.get(
            conversation_id
        )

        if (
            conversation is None
            or conversation["repository"] != normalized_repo_url
        ):
            conversation = {
                "repository": normalized_repo_url,
                "messages": [],
            }

            app.state.conversation_histories[
                conversation_id
            ] = conversation

        conversation_history = conversation["messages"]

        print(
            f"[TIMING] Repository setup: "
            f"{time.perf_counter() - start:.2f}s"
        )

        # Rewrite follow-up question using conversation history.
        stage = time.perf_counter()

        search_question = app.state.generator.rewrite_question(
            question,
            conversation_history,
        )

        print(
            f"[QUERY REWRITE] {question} -> {search_question}"
        )

        print(
            f"[TIMING] Query rewriting: "
            f"{time.perf_counter() - stage:.2f}s"
        )

        # Retrieve broader evidence for project-wide overview questions.
        stage = time.perf_counter()

        ranked_chunks = retrieve_relevant_chunks(
            context["hybrid"],
            app.state.reranker,
            question,
            search_question,
        )

        print(
            f"[TIMING] Retrieval and reranking: "
            f"{time.perf_counter() - stage:.2f}s"
        )

        # Generate answer using original question and history.
        stage = time.perf_counter()

        result = generate_answer_with_retry(
            app.state.generator,
            question,
            ranked_chunks,
            conversation_history,
        )

        # Save successful question and answer.
        conversation_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        conversation_history.append(
            {
                "role": "assistant",
                "content": result["answer"],
            }
        )

        conversation["messages"] = conversation_history[-12:]

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
            "conversation_id": conversation_id,
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
        import traceback

        error_message = str(exc)

        print("[ERROR] Question answering failed:")
        traceback.print_exc()

        print(f"[ERROR MESSAGE] {error_message}")

        if (
            "429" in error_message
            or "RESOURCE_EXHAUSTED" in error_message.upper()
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
            or "UNAVAILABLE" in error_message.upper()
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