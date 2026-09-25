from pathlib import Path
import chromadb

from .config import CHROMA_DIR
from .embeddings import embed_texts, embed_query


COLLECTION_NAME = "zepto_support"


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def index_documents(documents):
    """
    documents: list of dicts with keys:
      id, text, source
    """
    collection = get_collection()

    texts = [d["text"] for d in documents]
    embeddings = embed_texts(texts)

    collection.upsert(
        ids=[d["id"] for d in documents],
        documents=texts,
        metadatas=[{"source": d["source"]} for d in documents],
        embeddings=embeddings,
    )
    return collection.count()


def retrieve(query: str, top_k: int = 3):
    collection = get_collection()
    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=top_k,
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    ids = result.get("ids", [[]])[0]

    return [
        {
            "id": ids[i],
            "text": documents[i],
            "source": metadatas[i].get("source", ids[i]),
        }
        for i in range(len(documents))
    ]
