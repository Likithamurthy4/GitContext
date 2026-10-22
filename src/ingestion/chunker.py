
import ast
from dataclasses import dataclass
from hashlib import sha256

from src.ingestion.document import Document


@dataclass
class CodeChunk:
    chunk_id: str
    file_path: str
    language: str
    content: str
    start_line: int
    end_line: int


def _make_chunk(
    document: Document,
    content: str,
    start_line: int,
    end_line: int,
) -> CodeChunk:
    identity = (
        f"{document.file_path}:{start_line}:"
        f"{end_line}:{content}"
    )
    return CodeChunk(
        chunk_id=sha256(identity.encode("utf-8")).hexdigest(),
        file_path=document.file_path,
        language=document.language,
        content=content,
        start_line=start_line,
        end_line=end_line,
    )


def _line_chunks(
    document: Document,
    max_lines: int,
    overlap_lines: int,
    start_line: int = 1,
    end_line: int | None = None,
) -> list[CodeChunk]:
    lines = document.content.splitlines()
    end_line = min(end_line or len(lines), len(lines))

    if start_line > end_line or not lines:
        return []

    chunks = []
    start = start_line - 1
    stop = end_line

    while start < stop:
        end = min(start + max_lines, stop)
        content = "\n".join(lines[start:end])

        chunks.append(
            _make_chunk(document, content, start + 1, end)
        )

        if end == stop:
            break

        start = end - overlap_lines

    return chunks


def _ast_chunks(
    document: Document,
    max_lines: int,
    overlap_lines: int,
) -> list[CodeChunk]:
    source = document.content
    lines = source.splitlines()

    if not lines:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Incomplete or invalid Python: retain robust line chunking.
        return _line_chunks(document, max_lines, overlap_lines)

    if not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    ):
        return _line_chunks(document, max_lines, overlap_lines)

    # Build ranges for top-level statements. Each function or class
    # becomes a logical unit; other statements remain represented too.
    nodes = sorted(
        tree.body,
        key=lambda node: (node.lineno, node.end_lineno or node.lineno),
    )

    chunks = []
    cursor = 1

    for node in nodes:
        node_start = node.lineno
        node_end = node.end_lineno or node.lineno

        # Preserve module-level statements, comments and gaps before
        # each top-level AST node.
        if cursor < node_start:
            chunks.extend(
                _line_chunks(
                    document,
                    max_lines,
                    overlap_lines,
                    cursor,
                    node_start - 1,
                )
            )

        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            # Keep the whole logical unit when it fits.
            if node_end - node_start + 1 <= max_lines:
                content = "\n".join(lines[node_start - 1:node_end])
                chunks.append(
                    _make_chunk(
                        document, content, node_start, node_end
                    )
                )
            else:
                # Oversized functions/classes are split safely by lines.
                chunks.extend(
                    _line_chunks(
                        document,
                        max_lines,
                        overlap_lines,
                        node_start,
                        node_end,
                    )
                )
        else:
            # Preserve imports, assignments, and other module statements.
            chunks.extend(
                _line_chunks(
                    document,
                    max_lines,
                    overlap_lines,
                    node_start,
                    node_end,
                )
            )

        cursor = max(cursor, node_end + 1)

    # Preserve trailing comments or other source text.
    if cursor <= len(lines):
        chunks.extend(
            _line_chunks(
                document,
                max_lines,
                overlap_lines,
                cursor,
                len(lines),
            )
        )

    return chunks


def chunk_document(
    document: Document,
    max_lines: int = 80,
    overlap_lines: int = 10,
    strategy: str = "lines",
) -> list[CodeChunk]:
    if max_lines < 1:
        raise ValueError("max_lines must be at least 1")

    if overlap_lines < 0 or overlap_lines >= max_lines:
        raise ValueError(
            "overlap_lines must be >= 0 and less than max_lines"
        )

    if strategy not in {"lines", "ast"}:
        raise ValueError("strategy must be 'lines' or 'ast'")

    if strategy == "ast" and document.language.lower() in {
        "python", "py"
    }:
        return _ast_chunks(document, max_lines, overlap_lines)

    return _line_chunks(document, max_lines, overlap_lines)


def chunk_documents(
    documents: list[Document],
    max_lines: int = 80,
    overlap_lines: int = 10,
    strategy: str = "lines",
) -> list[CodeChunk]:
    chunks = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                max_lines=max_lines,
                overlap_lines=overlap_lines,
                strategy=strategy,
            )
        )

    return chunks
