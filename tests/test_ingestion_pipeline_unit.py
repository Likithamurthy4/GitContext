
from pathlib import Path

import pytest

from src.ingestion.document import Document
from src.ingestion.pipeline import ingest_repository


def test_ingests_supported_files(tmp_path, monkeypatch):
    file_path = tmp_path / "app.py"
    file_path.write_text("print('hello')", encoding="utf-8")

    monkeypatch.setattr(
        "src.ingestion.pipeline.scan_repository",
        lambda repo: [Path("app.py")],
    )

    monkeypatch.setattr(
        "src.ingestion.pipeline.read_file",
        lambda path: "print('hello')",
    )

    documents = ingest_repository(tmp_path)

    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].file_path == "app.py"
    assert documents[0].content == "print('hello')"
    assert documents[0].language == "python"


def test_maps_typescript_language(tmp_path, monkeypatch):
    (tmp_path / "app.ts").write_text("const x = 1;", encoding="utf-8")

    monkeypatch.setattr(
        "src.ingestion.pipeline.scan_repository",
        lambda repo: [Path("app.ts")],
    )
    monkeypatch.setattr(
        "src.ingestion.pipeline.read_file",
        lambda path: "const x = 1;",
    )

    documents = ingest_repository(tmp_path)

    assert documents[0].language == "typescript"


def test_returns_empty_list_when_no_files_found(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.ingestion.pipeline.scan_repository",
        lambda repo: [],
    )

    assert ingest_repository(tmp_path) == []


def test_skips_file_when_reading_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "src.ingestion.pipeline.scan_repository",
        lambda repo: [Path("broken.py")],
    )

    def fail_read(path):
        raise OSError("cannot read file")

    monkeypatch.setattr("src.ingestion.pipeline.read_file", fail_read)

    documents = ingest_repository(tmp_path)

    assert documents == []
    assert "Skipping broken.py" in capsys.readouterr().out
