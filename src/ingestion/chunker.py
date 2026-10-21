
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from src.ingestion.document import Document


@dataclass
class CodeChunk:
    chunk_id: str
    file_path: str
    language: str
    content: str
    start_line: int
    end_line: int


def chunk_document(
    document: Document,
    max_lines: int = 80,
    overlap_lines: int = 10,
) -> list[CodeChunk]:
    if max_lines < 1:
        raise ValueError("max_lines must be at least 1")

    if overlap_lines < 0 or overlap_lines >= max_lines:
        raise ValueError(
            "overlap_lines must be >= 0 and less than max_lines"
        )

    lines = document.content.splitlines()
    if not lines:
        return []

    chunks = []
    start = 0

    while start < len(lines):
        end = min(start + max_lines, len(lines))
        content = "\n".join(lines[start:end])

        identity = (
            f"{document.file_path}:{start + 1}:{end}:{content}"
        )
        chunk_id = sha256(identity.encode("utf-8")).hexdigest()

        chunks.append(
            CodeChunk(
                chunk_id=chunk_id,
                file_path=document.file_path,
                language=document.language,
                content=content,
                start_line=start + 1,
                end_line=end,
            )
        )

        if end == len(lines):
            break

        start = end - overlap_lines

    return chunks


def chunk_documents(
    documents: list[Document],
    max_lines: int = 80,
    overlap_lines: int = 10,
) -> list[CodeChunk]:
    chunks = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                max_lines=max_lines,
                overlap_lines=overlap_lines,
            )
        )

    return chunks
