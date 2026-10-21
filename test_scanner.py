
from pathlib import Path

from src.ingestion.file_scanner import scan_repository


repo_path = Path("data/repos/itsdangerous")

files = scan_repository(repo_path)

print(f"Total selected files: {len(files)}")
print("\nFirst 20 selected files:")

for file_path in files[:20]:
    print(file_path)
