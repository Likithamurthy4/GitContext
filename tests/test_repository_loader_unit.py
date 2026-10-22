
import subprocess

import pytest

from src.ingestion.repository_loader import clone_repository


def test_clones_repository_to_destination(tmp_path, monkeypatch):
    destination = tmp_path / "repos" / "example"

    def fake_run(command, **kwargs):
        assert command[:3] == ["git", "clone", "--depth"]
        assert command[3] == "1"
        assert command[4] == "https://github.com/example/repo"
        assert command[5] == str(destination)
        assert kwargs["check"] is True
        return None

    monkeypatch.setattr(
        "src.ingestion.repository_loader.subprocess.run",
        fake_run,
    )

    result = clone_repository(
        "https://github.com/example/repo",
        destination,
    )

    assert result == destination
    assert destination.parent.exists()


def test_existing_destination_raises_error(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        clone_repository("https://github.com/example/repo", destination)


def test_git_failure_is_propagated(tmp_path, monkeypatch):
    destination = tmp_path / "repo"

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git clone")

    monkeypatch.setattr(
        "src.ingestion.repository_loader.subprocess.run",
        fake_run,
    )

    with pytest.raises(subprocess.CalledProcessError):
        clone_repository("https://github.com/example/repo", destination)
