
from pathlib import Path

from src.ingestion.pipeline import ingest_repository


repo_path = Path("data/repos/itsdangerous")

documents = ingest_repository(repo_path)

print(f"Files successfully read: {len(documents)}")
print("\nFirst 10 documents:")

for document in documents[:10]:
    print(f"\nFile: {document.file_path}")
    print(f"Language: {document.language}")
    print(f"Characters: {len(document.content)}")
