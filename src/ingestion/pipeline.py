
from pathlib import Path

from src.ingestion.document import Document
from src.ingestion.file_reader import read_file
from src.ingestion.file_scanner import scan_repository


LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".php": "php",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
    ".md": "markdown",
    ".rst": "restructuredtext",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
}


def ingest_repository(repo_path: Path) -> list[Document]:
    """Scan a repository and create structured documents."""

    repo_path = repo_path.resolve()
    file_paths = scan_repository(repo_path)

    documents = []

    for relative_path in file_paths:
        full_path = repo_path / relative_path

        try:
            content = read_file(full_path)
        except (OSError, ValueError) as exc:
            print(f"Skipping {relative_path}: {exc}")
            continue

        document = Document(
            file_path=relative_path.as_posix(),
            content=content,
            language=LANGUAGE_MAP.get(
                full_path.suffix.lower(), "unknown"
            ),
        )

        documents.append(document)

    return documents
