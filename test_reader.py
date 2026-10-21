
from pathlib import Path

from src.ingestion.file_reader import read_file


file_path = Path(
    "data/repos/itsdangerous/src/itsdangerous/signer.py"
)

content = read_file(file_path)

print(f"File: {file_path}")
print(f"Characters read: {len(content)}")
print("\nFirst 15 lines:\n")

for line in content.splitlines()[:15]:
    print(line)
