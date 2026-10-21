
from pathlib import Path

from src.ingestion.repository_loader import clone_repository

repo_url = "https://github.com/pallets/itsdangerous.git"
destination = Path("data/repos/itsdangerous")

try:
    repo_path = clone_repository(repo_url, destination)
    print(f"Repository cloned successfully: {repo_path.resolve()}")
except Exception as exc:
    print(f"Repository cloning failed: {exc}")
