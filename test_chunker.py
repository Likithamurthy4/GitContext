
from pathlib import Path

from src.ingestion.chunker import chunk_documents
from src.ingestion.pipeline import ingest_repository


repo_path = Path("data/repos/itsdangerous")

documents = ingest_repository(repo_path)
chunks = chunk_documents(documents)

print(f"Documents: {len(documents)}")
print(f"Total chunks: {len(chunks)}")

for chunk in chunks[:5]:
    print(f"\nChunk ID: {chunk.chunk_id[:12]}")
    print(f"File: {chunk.file_path}")
    print(f"Language: {chunk.language}")
    print(f"Lines: {chunk.start_line}-{chunk.end_line}")
    print(f"Characters: {len(chunk.content)}")
    print(chunk.content[:200])
