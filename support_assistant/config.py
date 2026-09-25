from pathlib import Path
import os

PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parent

DOCS_DIR = PACKAGE_DIR / "docs"
CHROMA_DIR = PACKAGE_DIR / "chroma_db"

MOCK_LLM = os.getenv("MOCK_LLM", "1")