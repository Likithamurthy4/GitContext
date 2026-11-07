from types import SimpleNamespace

from google.genai import types
import pytest

from src.llm.answer_generator import (
    ANSWER_MAX_OUTPUT_TOKENS,
    AnswerGenerator,
    keep_cited_sources,
)


def test_keeps_only_valid_sources_referenced_by_answer():
    sources = [
        {
            "source_id": "S1",
            "file_path": "src/auth.py",
            "start_line": 1,
            "end_line": 4,
        },
        {
            "source_id": "S2",
            "file_path": "src/token.py",
            "start_line": 8,
            "end_line": 12,
        },
    ]

    answer, cited_sources = keep_cited_sources(
        "Tokens are checked here [S1, S2]. Missing [S9].",
        sources,
    )

    assert answer == "Tokens are checked here [S1, S2]. Missing."
    assert cited_sources == sources


def test_drops_source_metadata_when_answer_has_no_citations():
    sources = [
        {
            "source_id": "S1",
            "file_path": "src/auth.py",
            "start_line": 1,
            "end_line": 4,
        },
    ]

    answer, cited_sources = keep_cited_sources(
        "The repository provides an authentication helper.",
        sources,
    )

    assert answer == "The repository provides an authentication helper."
    assert cited_sources == []


def test_removes_invalid_citations_without_dropping_valid_references():
    sources = [
        {
            "source_id": "S1",
            "file_path": "src/auth.py",
            "start_line": 1,
            "end_line": 4,
        },
    ]

    answer, cited_sources = keep_cited_sources(
        "Valid reference [S1], invented reference [S3].",
        sources,
    )

    assert answer == "Valid reference [S1], invented reference."
    assert cited_sources == sources


def test_rewrite_question_uses_recent_conversation_history():
    class FakeModels:
        prompt = ""

        def generate_content(self, *, model, contents, config):
            self.prompt = contents
            return SimpleNamespace(text="Explain database_load.py")

    models = FakeModels()
    generator = AnswerGenerator.__new__(AnswerGenerator)
    generator.client = SimpleNamespace(models=models)
    generator.model = "test-model"
    history = [
        {"role": "user", "content": f"old question {index}"}
        if index % 2 == 0
        else {"role": "assistant", "content": f"old answer {index}"}
        for index in range(8)
    ]

    rewritten = generator.rewrite_question(
        "Which function does that?",
        history,
    )

    assert rewritten == "Explain database_load.py"
    assert "old question 0" not in models.prompt
    assert "old answer 1" not in models.prompt
    assert "old question 2" in models.prompt
    assert "old answer 7" in models.prompt
    assert "Which function does that?" in models.prompt


def test_generation_allows_complete_answers_and_reports_token_limit(capsys):
    class FakeModels:
        configs = []
        prompts = []

        def generate_content_stream(self, *, model, contents, config):
            self.configs.append(config)
            self.prompts.append(contents)
            is_continuation = len(self.configs) > 1
            return iter(
                [
                    SimpleNamespace(
                        text=(
                            " It validates the input [S1]."
                            if is_continuation
                            else "The repository supports this behavior."
                        ),
                        candidates=[
                            SimpleNamespace(
                                finish_reason=(
                                    types.FinishReason.STOP
                                    if is_continuation
                                    else types.FinishReason.MAX_TOKENS
                                )
                            )
                        ],
                    )
                ]
            )

    models = FakeModels()
    generator = AnswerGenerator.__new__(AnswerGenerator)
    generator.client = SimpleNamespace(models=models)
    generator.model = "test-model"

    result = generator.generate(
        "What does this do?",
        [
            {
                "file_path": "src/example.py",
                "start_line": 1,
                "end_line": 2,
                "language": "python",
                "content": "def example(): pass",
            }
        ],
        conversation_history=[
            {"role": "user", "content": f"question {index}"}
            if index % 2 == 0
            else {"role": "assistant", "content": f"answer {index}"}
            for index in range(15)
        ],
    )

    assert all(
        config.max_output_tokens == ANSWER_MAX_OUTPUT_TOKENS
        for config in models.configs
    )
    assert ANSWER_MAX_OUTPUT_TOKENS > 300
    assert result["answer"] == (
        "The repository supports this behavior. "
        "It validates the input [S1]."
    )
    assert len(models.configs) == 2
    assert "answer 3" in models.prompts[0]
    assert "question 14" in models.prompts[0]
    assert "question 0" not in models.prompts[0]
    assert "\nassistant: answer 1\n" not in models.prompts[0]
    assert "Already generated answer" in models.prompts[1]
    assert "The repository supports this behavior." in models.prompts[1]
    assert "[LLM] Continuing answer" in capsys.readouterr().out


def test_generation_does_not_return_answer_after_continuation_limit():
    class FakeModels:
        def generate_content_stream(self, *, model, contents, config):
            return iter(
                [
                    SimpleNamespace(
                        text="Still incomplete",
                        candidates=[
                            SimpleNamespace(
                                finish_reason=types.FinishReason.MAX_TOKENS
                            )
                        ],
                    )
                ]
            )

    generator = AnswerGenerator.__new__(AnswerGenerator)
    generator.client = SimpleNamespace(models=FakeModels())
    generator.model = "test-model"

    with pytest.raises(RuntimeError, match="exceeded the model output limit"):
        generator.generate(
            "Explain everything",
            [
                {
                    "file_path": "src/example.py",
                    "start_line": 1,
                    "end_line": 1,
                    "language": "python",
                    "content": "pass",
                }
            ],
        )
