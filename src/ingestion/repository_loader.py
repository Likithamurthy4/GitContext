import subprocess
from pathlib import Path


def clone_repository(repo_url: str, destination: Path) -> Path:
    """Clone a Git repository into a local destination."""

    if destination.exists():
        raise FileExistsError(
            f"Destination already exists: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(destination)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    return destination