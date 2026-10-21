
from pathlib import Path


SUPPORTED_TEXT_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".c", ".cpp", ".h", ".hpp",
    ".go", ".rs", ".php", ".html", ".css",
    ".sql", ".md", ".rst", ".txt", ".json",
    ".yaml", ".yml", ".toml",
}


def read_file(file_path: Path) -> str:
    """Read a supported text file and return its contents."""

    if not file_path.is_file():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if file_path.suffix.lower() not in SUPPORTED_TEXT_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}"
        )

    return file_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
