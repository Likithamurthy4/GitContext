
from sentence_transformers import CrossEncoder


class CodeReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        limit: int = 5,
    ) -> list[dict]:
        if not candidates or limit <= 0:
            return []

        pairs = [
            (
                query,
                f"File: {item['file_path']}\n"
                f"Language: {item['language']}\n"
                f"Code:\n{item['content']}",
            )
            for item in candidates
        ]

        scores = self.model.predict(pairs)

        ranked = []
        for item, score in zip(candidates, scores):
            ranked.append({
                **item,
                "rerank_score": float(score),
            })

        return sorted(
            ranked,
            key=lambda item: item["rerank_score"],
            reverse=True,
        )[:limit]
