import os
from config.config import CHROMA_DIR, RETRIEVAL_TOP_K

_COLLECTION_NAME = "krishisahayak_schemes"

def build_vector_store(chunks: list, embedding_model):
    try:
        from langchain_chroma import Chroma
        os.makedirs(CHROMA_DIR, exist_ok=True)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name=_COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
        )
        print(f"[VectorStore] Built index with {len(chunks)} chunks → '{CHROMA_DIR}'")
        return vectorstore
    except Exception as e:
        raise RuntimeError(f"[VectorStore] Failed to build index: {e}") from e

def load_vector_store(embedding_model):
    try:
        from langchain_chroma import Chroma

        if not os.path.exists(CHROMA_DIR):
            return None

        vectorstore = Chroma(
            collection_name=_COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=CHROMA_DIR,
        )

        # Guard against an uninitialised (empty) collection via public API.
        sample = vectorstore.get(limit=1, include=["metadatas"])
        ids = sample.get("ids", []) if isinstance(sample, dict) else []
        if not ids:
            return None

        print(f"[VectorStore] Loaded existing index from '{CHROMA_DIR}'")
        return vectorstore
    except Exception as e:
        print(f"[VectorStore] Could not load index: {e}")
        return None

def retrieve_context(query: str, vectorstore, top_k: int = RETRIEVAL_TOP_K) -> str:
    try:
        results = vectorstore.similarity_search(query, k=top_k)
        if not results:
            return ""
        parts = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "knowledge base")
            parts.append(
                f"[Excerpt {i} — {os.path.basename(source)}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        print(f"[VectorStore] Retrieval error: {e}")
        return ""

def _format_doc_key(doc) -> str:
    try:
        source = doc.metadata.get("source", "knowledge base")
        head = (doc.page_content or "")[:220]
        return f"{source}::{head}"
    except Exception:
        return str(id(doc))

def retrieve_context_hybrid(
    query: str,
    vectorstore,
    kb_chunks: list,
    top_k: int = RETRIEVAL_TOP_K,
) -> dict:
    try:
        from langchain_community.retrievers import BM25Retriever

        if vectorstore is None and not kb_chunks:
            return {"context": "", "citations": []}

        vector_docs = []
        lexical_docs = []

        # Dense retrieval
        if vectorstore is not None:
            vector_docs = vectorstore.similarity_search(query, k=top_k)

        # Lexical retrieval
        if kb_chunks:
            bm25 = BM25Retriever.from_documents(kb_chunks)
            bm25.k = top_k
            lexical_docs = bm25.invoke(query)

        # Reciprocal Rank Fusion to combine both rankings
        fused = {}
        rrf_k = 60

        for rank, doc in enumerate(vector_docs, 1):
            key = _format_doc_key(doc)
            fused.setdefault(key, {"doc": doc, "score": 0.0})
            fused[key]["score"] += 1.0 / (rrf_k + rank)

        for rank, doc in enumerate(lexical_docs, 1):
            key = _format_doc_key(doc)
            fused.setdefault(key, {"doc": doc, "score": 0.0})
            fused[key]["score"] += 1.0 / (rrf_k + rank)

        ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        docs = [item["doc"] for item in ranked]

        if not docs:
            return {"context": "", "citations": []}

        parts = []
        citations = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "knowledge base")
            source_name = os.path.basename(source)
            parts.append(f"[Excerpt {i} - {source_name}]\n{doc.page_content}")
            if source_name not in citations:
                citations.append(source_name)

        return {
            "context": "\n\n---\n\n".join(parts),
            "citations": citations,
        }
    except Exception as e:
        print(f"[VectorStore] Hybrid retrieval error: {e}")
        # Safe fallback to existing dense-only retrieval
        context = retrieve_context(query, vectorstore, top_k=top_k) if vectorstore else ""
        return {"context": context, "citations": []}
