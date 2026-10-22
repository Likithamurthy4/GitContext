
from src.ingestion.document import Document


def test_document_stores_fields():
    document = Document(
        file_path="src/example.py",
        content="print('Hello')",
        language="python",
    )

    assert document.file_path == "src/example.py"
    assert document.content == "print('Hello')"
    assert document.language == "python"
