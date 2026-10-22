
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
