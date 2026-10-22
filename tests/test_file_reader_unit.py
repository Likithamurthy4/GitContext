
from pathlib import Path

import pytest

from src.ingestion.file_reader import read_file


def test_reads_supported_text_file(tmp_path):
    file_path = tmp_path / "example.py"
    file_path.write_text("print('hello')", encoding="utf-8")

    assert read_file(file_path) == "print('hello')"


def test_reads_uppercase_extension(tmp_path):
    file_path = tmp_path / "README.MD"
    file_path.write_text("# GitContext", encoding="utf-8")

    assert read_file(file_path) == "# GitContext"


def test_missing_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_file(tmp_path / "missing.py")


def test_unsupported_extension_raises_error(tmp_path):
    file_path = tmp_path / "program.exe"
    file_path.write_text("not supported", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        read_file(file_path)


def test_invalid_utf8_is_replaced(tmp_path):
    file_path = tmp_path / "example.py"
    file_path.write_bytes(b"hello \xff world")

    content = read_file(file_path)

    assert "hello" in content
    assert "\ufffd" in content
