
from src.ingestion.document import Document


document = Document(
    file_path="src/itsdangerous/signer.py",
    content="class SigningAlgorithm: pass",
    language="python",
)

print(document)
print(document.file_path)
print(document.language)
