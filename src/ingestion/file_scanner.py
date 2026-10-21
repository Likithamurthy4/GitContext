
from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".c", ".cpp", ".h", ".hpp",
    ".go", ".rs", ".php", ".html", ".css",
    ".sql", ".md", ".rst", ".txt", ".json",
    ".yaml", ".yml", ".toml",
}

IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    "dist",
    "build",
}

MAX_FILE_SIZE_BYTES = 1_000_000  # 1 MB


def scan_repository(repo_path: Path) -> list[Path]:
    """Return supported files while skipping ignored folders and large files."""

    repo_path = repo_path.resolve()

    if not repo_path.is_dir():
        raise NotADirectoryError(
            f"Repository directory not found: {repo_path}"
        )

    selected_files = []

    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(repo_path)

        if any(
            part in IGNORED_DIRECTORIES
            for part in relative_path.parts[:-1]
        ):
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
            continue

        selected_files.append(relative_path)

    return sorted(selected_files)
