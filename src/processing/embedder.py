import os
import uuid
from collections.abc import Mapping

# Prevent tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

from src.core import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)

client = PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


def embed_chunks_with_pages(chunks_with_pages: list[dict], source_id: str) -> None:
    """
    Embed chunks that include page information.

    Args:
        chunks_with_pages: List of dicts with 'text', 'pages', 'chunk_id' keys
        source_id: Source document identifier
    """
    # 🧼 Optional: delete old entries for the same source
    existing = collection.get(where={"source": source_id}) or {"ids": []}
    if existing and existing["ids"]:
        collection.delete(ids=existing["ids"])

    # Extract text for embedding
    texts = [chunk["text"] for chunk in chunks_with_pages]
    embeddings = model.encode(texts).tolist()

    namespace_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
    ids = [
        str(uuid.uuid5(namespace_uuid, f"{source_id}_{chunk['chunk_id']}"))
        for chunk in chunks_with_pages
    ]

    # Create metadata including page information
    metadata: list[Mapping[str, str | int | float | bool | None]] = []
    for i, chunk in enumerate(chunks_with_pages):
        meta = {
            "source": source_id,
            "chunk_index": i,
            "chunk_id": chunk["chunk_id"],
            "pages": str(chunk["pages"]),  # Store as string for ChromaDB compatibility
            "page_count": len(chunk["pages"])
        }
        metadata.append(meta)

    collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadata)
    print(f"✅ Stored {len(chunks_with_pages)} page-aware chunks into ChromaDB as '{source_id}'")


def embed_chunks(chunks: list[str], source_id: str) -> None:
    """Legacy function for backward compatibility with string-only chunks."""
    # 🧼 Optional: delete old entries for the same source
    existing = collection.get(where={"source": source_id}) or {"ids": []}
    if existing and existing["ids"]:
        collection.delete(ids=existing["ids"])

    embeddings = model.encode(chunks).tolist()
    namespace_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
    ids = [
        str(uuid.uuid5(namespace_uuid, f"{source_id}_{i}")) for i in range(len(chunks))
    ]
    metadata: list[Mapping[str, str | int | float | bool | None]] = [
        {"source": source_id, "chunk_index": i} for i in range(len(chunks))
    ]

    collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadata)
    # print(f"✅ Stored {len(chunks)} chunks into ChromaDB as '{source_id}'")


def get_all_chunks_for_document(source_id: str) -> list[dict]:
    """
    Get all chunks for a document without using vector search.
    Returns chunks sorted by chunk_index to maintain document order.
    """
    # Get all chunks for the source document
    results = collection.get(
        where={"source": source_id}, include=["documents", "metadatas"]
    )

    if not results or not results["documents"]:
        return []

    # Create list of chunks with metadata
    chunks = []
    metadatas = results.get("metadatas") or []
    for doc, meta in zip(results["documents"], metadatas, strict=False):
        chunk_info = {
            "text": doc,
            "chunk_index": meta.get("chunk_index", 0),
            "source": meta.get("source"),
        }

        # Add page information if available
        if meta.get("pages"):
            try:
                # Parse pages from string format
                import ast
                pages = ast.literal_eval(meta["pages"])
                chunk_info["pages"] = pages
                chunk_info["chunk_id"] = meta.get("chunk_id")
                chunk_info["page_count"] = meta.get("page_count", len(pages))
            except (ValueError, SyntaxError):
                # Fallback for malformed page data
                chunk_info["pages"] = []

        chunks.append(chunk_info)

    # Sort by chunk_index to maintain document order
    chunks.sort(
        key=lambda x: (
            int(x["chunk_index"]) if isinstance(x["chunk_index"], int | str) else 0
        )
    )

    return chunks


def query_chunks(
    query: str, top_k: int = 5, source_id: str | None = None
) -> list[dict]:
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

    if not results or not results.get("documents") or not results["documents"]:
        return []

    docs = (
        results["documents"][0]
        if results.get("documents") and results["documents"]
        else []
    )
    distances = (
        results["distances"][0]
        if results.get("distances") and results["distances"]
        else []
    )
    metadatas = (
        results["metadatas"][0]
        if results.get("metadatas") and results["metadatas"]
        else []
    )

    results_list = []
    for doc, score, meta in zip(docs, distances, metadatas, strict=False):
        result = {
            "text": doc,
            "score": round(score, 4),
            "chunk_index": meta.get("chunk_index") if isinstance(meta, dict) else 0,
            "source": meta.get("source") if isinstance(meta, dict) else None,
        }

        # Add page information if available
        if isinstance(meta, dict) and "pages" in meta and meta["pages"]:
            try:
                import ast
                pages = ast.literal_eval(meta["pages"])
                result["pages"] = pages
                result["chunk_id"] = meta.get("chunk_id")
                result["page_count"] = meta.get("page_count", len(pages))
            except (ValueError, SyntaxError):
                result["pages"] = []

        results_list.append(result)

    return results_list
