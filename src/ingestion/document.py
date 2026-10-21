
from dataclasses import dataclass


@dataclass
class Document:
    file_path: str
    content: str
    language: str
