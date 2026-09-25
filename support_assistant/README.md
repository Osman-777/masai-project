# Support Assistant

A small Zepto policy RAG support assistant using:

- LangGraph
- ChromaDB
- `all-MiniLM-L6-v2`
- FastAPI
- Pydantic
- Deterministic `MOCK_LLM` baseline

## Expected project structure

Place this package inside the project root:

```text
project/
├── docs/
│   ├── doc_01.txt
│   ├── ...
│   └── doc_08.txt
├── support_assistant/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── prompts.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── ingest.py
│   ├── graph.py
│   └── api.py
└── main.py
```

## Install

```bash
pip install fastapi uvicorn langgraph chromadb sentence-transformers pydantic
```

## Ingest the eight policy documents

From the project root:

```bash
python -m support_assistant.ingest
```

This creates/updates the persistent ChromaDB collection.

## Run the API

```bash
uvicorn support_assistant.api:app --host 0.0.0.0 --port 7860
```

## Test

```bash
curl -X POST http://127.0.0.1:7860/ask   -H "Content-Type: application/json"   -d '{"query":"What is the delivery fee for orders below INR 149?"}'
```

The default path is deterministic and does not require an LLM API key.
