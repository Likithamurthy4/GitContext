
import json
import statistics
from pathlib import Path

from src.ingestion.chunker import chunk_documents
from src.ingestion.pipeline import ingest_repository
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import CodeReranker
from src.retrieval.vector_store import VectorStore


ROOT = Path(__file__).resolve().parent.parent
CASES_FILE = ROOT / "evaluation" / "retrieval_cases.json"
REPORT_FILE = ROOT / "evaluation" / "baseline_report.json"
DB_PATH = ROOT / "data" / "qdrant" / "evaluation"

TOP_K = 5
CANDIDATE_LIMIT = 15


def reciprocal_rank(
    results: list[dict],
    expected_files: set[str],
) -> float:
    ranked_files = list(
        dict.fromkeys(result["file_path"] for result in results)
    )

    for rank, file_path in enumerate(ranked_files, start=1):
        if file_path in expected_files:
            return 1.0 / rank

    return 0.0

def evaluate_stage(
    cases: list[dict],
    search_function,
) -> dict:
    hits = 0
    reciprocal_ranks = []
    case_results = []

    for case in cases:
        expected = set(case["expected_files"])
        results = search_function(case["question"])
        found_files = list(
            dict.fromkeys(item["file_path"] for item in results)
        )

        hit = any(path in expected for path in found_files[:TOP_K])
        hits += int(hit)

        rr = reciprocal_rank(results, expected)
        reciprocal_ranks.append(rr)

        case_results.append({
            "question": case["question"],
            "expected_files": sorted(expected),
            "retrieved_files": found_files,
            "hit_at_5": hit,
            "reciprocal_rank": rr,
        })

    total = len(cases)

    return {
        "case_count": total,
        "hit_rate_at_5": hits / total if total else 0.0,
        "mrr": (
            statistics.mean(reciprocal_ranks)
            if reciprocal_ranks else 0.0
        ),
        "cases": case_results,
    }


def main() -> None:
    with CASES_FILE.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not cases:
        raise ValueError("The evaluation dataset is empty.")

    print("[EVAL] Reading local GitContext source files...")
    documents = ingest_repository(
        ROOT,
        extra_ignored_directories={
            "data",
            "evaluation",
        },
    )
    
    if not documents:
        raise RuntimeError("No supported source files were found.")

    chunks = chunk_documents(documents, strategy="ast")

    print("\n--- VECTOR STORE CHUNKS ---")

    for chunk in chunks:
        if chunk.file_path.replace("\\", "/").endswith(
            "src/retrieval/vector_store.py"
        ):
            print(
                f"\nLines {chunk.start_line}-{chunk.end_line}"
            )
            print(chunk.content[:1200])

    if not chunks:
        raise RuntimeError("No code chunks were generated.")

    print(f"[EVAL] Loaded {len(documents)} files.")
    print(f"[EVAL] Created {len(chunks)} chunks.")

    vector_store = None

    try:
        vector_store = VectorStore(
            storage_path=str(ROOT / "data" / "qdrant" / "evaluation_v2"),
            collection_name="gitcontext_evaluation",
        )

        print("[EVAL] Indexing code for semantic search...")
        vector_store.index_chunks(chunks)

        hybrid = HybridSearch(vector_store)
        hybrid.build_index(chunks)
        target_query = "How does semantic search encode a query before searching?"

        print("\n--- TARGET CHUNK CHECK ---")

        for chunk in chunks:
            if (
                chunk.file_path.replace("\\", "/").endswith(
                    "src/retrieval/vector_store.py"
                )
                and "query_vector" in chunk.content
            ):
                print("File:", chunk.file_path)
                print("Lines:", chunk.start_line, "-", chunk.end_line)
                print("Contains model.encode:", "self.model.encode" in chunk.content)
                print("Contains query_points:", "query_points" in chunk.content)
                print("Contains query_vector:", "query_vector" in chunk.content)
                print("Relevant code:")
                print(chunk.content[-1800:])

        print("\n--- DIRECT SEMANTIC SEARCH CHECK ---")

        results = vector_store.search(target_query, limit=30)

        for rank, result in enumerate(results, start=1):
            if result["file_path"].replace("\\", "/").endswith(
                "src/retrieval/vector_store.py"
            ):
                print("VectorStore found at semantic rank:", rank)
                print("Score:", result["score"])
                print("Chunk lines:", result.get("start_line"), "-", result.get("end_line"))

        diagnostic_question = (
            "How does semantic search encode a query before searching?"
        )

        diagnostic_results = hybrid.search(
            diagnostic_question,
            limit=CANDIDATE_LIMIT,
            candidate_limit=CANDIDATE_LIMIT,
        )

        print("\n[EVAL] Diagnostic retrieval results:")
        for rank, result in enumerate(diagnostic_results, start=1):
            print(
                f"{rank}. {result['file_path']} "
                f"(hybrid_score={result.get('hybrid_score', 0):.5f})"
            )

        print("[EVAL] Evaluating hybrid retrieval...")
        hybrid_report = evaluate_stage(
            cases,
            lambda question: hybrid.search(
                question,
                limit=TOP_K,
                candidate_limit=CANDIDATE_LIMIT,
            ),
        )

        print("[EVAL] Loading CrossEncoder reranker...")
        reranker = CodeReranker()

        print("[EVAL] Evaluating reranked results...")
        reranked_report = evaluate_stage(
            cases,
            lambda question: reranker.rerank(
                question,
                hybrid.search(
                    question,
                    limit=CANDIDATE_LIMIT,
                    candidate_limit=CANDIDATE_LIMIT,
                ),
                limit=TOP_K,
            ),
        )

        report = {
            "repository": str(ROOT),
            "dataset_size": len(cases),
            "top_k": TOP_K,
            "hybrid_retrieval": hybrid_report,
            "after_reranking": reranked_report,
        }

        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        print("\n=== GitContext Retrieval Evaluation ===")
        print(f"Questions: {len(cases)}")
        print(
            "Hybrid Hit Rate@5: "
            f"{hybrid_report['hit_rate_at_5']:.1%}"
        )
        print(f"Hybrid MRR: {hybrid_report['mrr']:.3f}")
        print(
            "Reranked Hit Rate@5: "
            f"{reranked_report['hit_rate_at_5']:.1%}"
        )
        print(f"Reranked MRR: {reranked_report['mrr']:.3f}")
        print(f"\nFull report saved to: {REPORT_FILE}")

    finally:
        if vector_store is not None:
            vector_store.close()


if __name__ == "__main__":
    main()
