
import pytest

from src.ingestion.document import Document
from src.ingestion.chunker import chunk_document, chunk_documents


def test_chunk_document_splits_lines():
    document = Document(
        file_path="example.py",
        content="line1\nline2\nline3\nline4\nline5",
        language="python",
    )

    chunks = chunk_document(
        document,
        max_lines=3,
        overlap_lines=1,
    )

    assert len(chunks) == 2
    assert chunks[0].content == "line1\nline2\nline3"
    assert chunks[1].content == "line3\nline4\nline5"


def test_chunk_line_numbers_are_correct():
    document = Document(
        file_path="example.py",
        content="a\nb\nc\nd",
        language="python",
    )

    chunks = chunk_document(
        document,
        max_lines=3,
        overlap_lines=1,
    )

    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 3
    assert chunks[1].start_line == 3
    assert chunks[1].end_line == 4


def test_empty_document_returns_no_chunks():
    document = Document(
        file_path="empty.py",
        content="",
        language="python",
    )

    assert chunk_document(document) == []


def test_invalid_max_lines_raises_error():
    document = Document(
        file_path="example.py",
        content="hello",
        language="python",
    )

    with pytest.raises(ValueError):
        chunk_document(document, max_lines=0)


def test_invalid_overlap_raises_error():
    document = Document(
        file_path="example.py",
        content="hello",
        language="python",
    )

    with pytest.raises(ValueError):
        chunk_document(document, max_lines=3, overlap_lines=3)


def test_chunk_ids_are_deterministic():
    document = Document(
        file_path="example.py",
        content="a\nb\nc",
        language="python",
    )

    first = chunk_document(document)
    second = chunk_document(document)

    assert [chunk.chunk_id for chunk in first] == [
        chunk.chunk_id for chunk in second
    ]


def test_chunk_documents_handles_multiple_documents():
    documents = [
        Document("one.py", "print(1)", "python"),
        Document("two.py", "print(2)", "python"),
    ]

    chunks = chunk_documents(documents)

    assert len(chunks) == 2
    assert {chunk.file_path for chunk in chunks} == {
        "one.py",
        "two.py",
    }

def test_ast_strategy_keeps_function_as_one_chunk():
    document = Document(
        file_path="example.py",
        content=(
            "def greet(name):\n"
            "    message = 'Hello ' + name\n"
            "    return message\n"
        ),
        language="python",
    )

    chunks = chunk_document(document, strategy="ast")

    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 3
    assert "def greet(name):" in chunks[0].content


def test_ast_strategy_preserves_class_and_function_names():
    document = Document(
        file_path="example.py",
        content=(
            "class Greeter:\n"
            "    def greet(self):\n"
            "        return 'Hello'\n"
            "\n"
            "def farewell():\n"
            "    return 'Bye'\n"
        ),
        language="python",
    )

    chunks = chunk_document(document, strategy="ast")

    assert any("class Greeter:" in c.content for c in chunks)
    assert any("def farewell():" in c.content for c in chunks)


def test_ast_strategy_preserves_module_level_statements():
    document = Document(
        file_path="example.py",
        content=(
            "import os\n"
            "\n"
            "def greet():\n"
            "    return os.getcwd()\n"
            "\n"
            "VALUE = 42\n"
        ),
        language="python",
    )

    chunks = chunk_document(document, strategy="ast")
    combined = "\n".join(c.content for c in chunks)

    assert "import os" in combined
    assert "def greet():" in combined
    assert "VALUE = 42" in combined


def test_ast_strategy_falls_back_for_invalid_python():
    document = Document(
        file_path="broken.py",
        content="line1\nline2\nline3\nline4",
        language="python",
    )

    chunks = chunk_document(
        document,
        max_lines=2,
        overlap_lines=0,
        strategy="ast",
    )

    assert len(chunks) == 2
    assert chunks[0].content == "line1\nline2"
    assert chunks[1].content == "line3\nline4"


def test_ast_strategy_splits_oversized_function():
    document = Document(
        file_path="large.py",
        content=(
            "def large():\n"
            "    first = 1\n"
            "    second = 2\n"
            "    third = 3\n"
            "    return first + second + third\n"
        ),
        language="python",
    )

    chunks = chunk_document(
        document,
        max_lines=3,
        overlap_lines=1,
        strategy="ast",
    )

    assert len(chunks) >= 2
    assert all(
        c.end_line - c.start_line + 1 <= 3
        for c in chunks
    )


def test_invalid_chunking_strategy_raises_error():
    document = Document(
        file_path="example.py",
        content="print('hello')",
        language="python",
    )

    with pytest.raises(ValueError):
        chunk_document(document, strategy="unknown")


def test_ast_strategy_preserves_chunk_metadata():
    document = Document(
        file_path="src/example.py",
        content="def greet():\n    return 'hi'\n",
        language="python",
    )

    chunks = chunk_document(document, strategy="ast")

    assert chunks[0].file_path == "src/example.py"
    assert chunks[0].language == "python"
    assert chunks[0].chunk_id