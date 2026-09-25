from pathlib import Path

from .config import DOCS_DIR
from .vectorstore import index_documents


def load_documents(docs_dir: Path = DOCS_DIR):
    documents = []

    for path in sorted(docs_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append({
                "id": path.stem,
                "text": text,
                "source": path.name,
            })

    if not documents:
        raise FileNotFoundError(
            f"No .txt documents found in {docs_dir}. "
            "Create docs/doc_01.txt ... docs/doc_08.txt first."
        )

    return documents


def ingest():
    documents = load_documents()
    return index_documents(documents)


if __name__ == "__main__":
    print(f"Indexed {ingest()} documents/chunks.")
