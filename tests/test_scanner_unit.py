
from pathlib import Path

import pytest

from src.ingestion.file_scanner import (
    MAX_FILE_SIZE_BYTES,
    scan_repository,
)


def test_scans_supported_files(tmp_path):
    (tmp_path / "app.py").write_text("print('hello')", encoding="utf-8")
    (tmp_path / "styles.css").write_text("body {}", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"fake image")

    files = scan_repository(tmp_path)

    assert files == [Path("app.py"), Path("styles.css")]


def test_ignores_unsupported_files(tmp_path):
    (tmp_path / "app.py").write_text("code", encoding="utf-8")
    (tmp_path / "notes.exe").write_text("ignored", encoding="utf-8")

    files = scan_repository(tmp_path)

    assert Path("app.py") in files
    assert Path("notes.exe") not in files


def test_ignores_excluded_directories(tmp_path):
    (tmp_path / "app.py").write_text("code", encoding="utf-8")

    ignored_dir = tmp_path / "node_modules"
    ignored_dir.mkdir()
    (ignored_dir / "dependency.js").write_text("code", encoding="utf-8")

    files = scan_repository(tmp_path)

    assert Path("app.py") in files
    assert Path("node_modules/dependency.js") not in files


def test_ignores_files_over_size_limit(tmp_path):
    (tmp_path / "large.py").write_bytes(
        b"x" * (MAX_FILE_SIZE_BYTES + 1)
    )
    (tmp_path / "small.py").write_text("x", encoding="utf-8")

    files = scan_repository(tmp_path)

    assert Path("small.py") in files
    assert Path("large.py") not in files


def test_returns_sorted_relative_paths(tmp_path):
    (tmp_path / "z.py").write_text("z", encoding="utf-8")
    (tmp_path / "a.py").write_text("a", encoding="utf-8")

    files = scan_repository(tmp_path)

    assert files == [Path("a.py"), Path("z.py")]


def test_missing_directory_raises_error(tmp_path):
    missing = tmp_path / "missing"

    with pytest.raises(NotADirectoryError):
        scan_repository(missing)


def test_scans_nested_supported_files(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "main.py").write_text("pass", encoding="utf-8")

    files = scan_repository(tmp_path)

    assert files == [Path("src/main.py")]
