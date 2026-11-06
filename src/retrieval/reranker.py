
from sentence_transformers import CrossEncoder


class CodeReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rank_constant: int = 60,
        hybrid_weight: float = 1.0,
        reranker_weight: float = 1.0,
    ):
        self.model = CrossEncoder(model_name)
        self.rank_constant = rank_constant
        self.hybrid_weight = hybrid_weight
        self.reranker_weight = reranker_weight

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

        # Preserve the original hybrid ranking.
        hybrid_ranked = sorted(
            enumerate(candidates),
            key=lambda pair: pair[1].get(
                "hybrid_score",
                pair[1].get("score", 0.0),
            ),
            reverse=True,
        )

        hybrid_ranks = {
            index: rank
            for rank, (index, _) in enumerate(
                hybrid_ranked, start=1
            )
        }

        # Rank candidates by CrossEncoder relevance.
        reranked_indices = sorted(
            range(len(candidates)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )
        reranker_ranks = {
            index: rank
            for rank, index in enumerate(
                reranked_indices, start=1
            )
        }

        ranked = []
        for index, item in enumerate(candidates):
            hybrid_rank = hybrid_ranks[index]
            reranker_rank = reranker_ranks[index]

            fusion_score = (
                self.hybrid_weight
                / (self.rank_constant + hybrid_rank)
                + self.reranker_weight
                / (self.rank_constant + reranker_rank)
            )

            ranked.append({
                **item,
                "rerank_score": float(scores[index]),
                "hybrid_rank": hybrid_rank,
                "reranker_rank": reranker_rank,
                "fusion_score": fusion_score,
            })

        return sorted(
            ranked,
            key=lambda item: item["fusion_score"],
            reverse=True,
        )[:limit]
