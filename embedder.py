import os
import uuid

# Prevent tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)

client = PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


def embed_chunks(chunks: list[str], source_id: str) -> None:
    # 🧼 Optional: delete old entries for the same source
    existing = collection.get(where={"source": source_id}) or {"ids": []}
    if existing and existing["ids"]:
        collection.delete(ids=existing["ids"])

    embeddings = model.encode(chunks).tolist()
    NAMESPACE_UUID = uuid.UUID("12345678-1234-5678-1234-567812345678")
    ids = [
        str(uuid.uuid5(NAMESPACE_UUID, f"{source_id}_{i}")) for i in range(len(chunks))
    ]
    metadata = [{"source": source_id, "chunk_index": i} for i in range(len(chunks))]

    collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadata)
    # print(f"✅ Stored {len(chunks)} chunks into ChromaDB as '{source_id}'")


def query_chunks(query: str, top_k: int = 5, source_id: str = None) -> list[dict]:
    query_embedding = model.encode([query])[0].tolist()

    if source_id:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"source": source_id},
            include=["documents", "distances", "metadatas"],
        )
    else:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )

    return [
        {
            "text": doc,
            "score": round(score, 4),
            "chunk_index": meta.get("chunk_index"),
            "source": meta.get("source"),
        }
        for doc, score, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
            strict=False,
        )
    ]
