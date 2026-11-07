
from sentence_transformers import CrossEncoder
from transformers import PreTrainedTokenizerBase


MAX_LENGTH = 512


def _candidate_windows(
    query: str,
    candidate_text: str,
    tokenizer: PreTrainedTokenizerBase,
) -> list[str]:
    """Split candidate text into bounded windows for reranking."""

    query_tokens = tokenizer.encode(
        query,
        add_special_tokens=False,
        truncation=True,
        max_length=128,
    )

    special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
    available_tokens = MAX_LENGTH - len(query_tokens) - special_tokens

    if available_tokens <= 0:
        query_tokens = query_tokens[:64]
        available_tokens = (
            MAX_LENGTH
            - len(query_tokens)
            - special_tokens
        )

    candidate_tokens = tokenizer.encode(
        candidate_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    if len(candidate_tokens) <= available_tokens:
        return [candidate_text]

    window_size = max(1, available_tokens - 16)
    step = max(1, window_size * 3 // 4)

    windows = []

    for start in range(0, len(candidate_tokens), step):
        window_tokens = candidate_tokens[start:start + window_size]

        window_text = tokenizer.decode(
            window_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        windows.append(window_text)

        if start + window_size >= len(candidate_tokens):
            break

    return windows


class CodeReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rank_constant: int = 60,
        hybrid_weight: float = 1.0,
        reranker_weight: float = 1.0,
    ):
        self.model = CrossEncoder(
            model_name,
            max_length=MAX_LENGTH,
        )

        self.model.tokenizer.model_max_length = MAX_LENGTH

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

        pairs = []
        pair_candidate_indices = []

        for index, item in enumerate(candidates):
            candidate_text = (
                f"File: {item['file_path']}\n"
                f"Language: {item['language']}\n"
                f"Code:\n{item['content']}"
            )

            windows = _candidate_windows(
                query,
                candidate_text,
                self.model.tokenizer,
            )

            for window in windows:
                pairs.append((query, window))
                pair_candidate_indices.append(index)

        window_scores = self.model.predict(
            pairs,
            batch_size=8,
        )

        scores = [float("-inf")] * len(candidates)

        for index, score in zip(
            pair_candidate_indices,
            window_scores,
        ):
            scores[index] = max(
                scores[index],
                float(score),
            )

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
                hybrid_ranked,
                start=1,
            )
        }

        reranked_indices = sorted(
            range(len(candidates)),
            key=lambda index: scores[index],
            reverse=True,
        )

        reranker_ranks = {
            index: rank
            for rank, index in enumerate(
                reranked_indices,
                start=1,
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
                "rerank_score": scores[index],
                "hybrid_rank": hybrid_rank,
                "reranker_rank": reranker_rank,
                "fusion_score": fusion_score,
            })

        return sorted(
            ranked,
            key=lambda item: item["rerank_score"],
            reverse=True,
        )[:limit]
