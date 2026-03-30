# ABOUTME: Phase 4 — embeds chunks into ChromaDB and builds hybrid retrieval (vector + BM25).
# ABOUTME: Produces a searchable ChromaDB collection and BM25 index for the RAG pipeline.

import json

from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun  # noqa: F401
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever  # noqa: F401
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder  # noqa: F401

from src.config import (
    CHUNKS_PATH,
    CHROMA_DIR,
    DEFAULT_BM25_WEIGHT,
    DEFAULT_FETCH_K,
    DEFAULT_K,
    DEFAULT_VECTOR_WEIGHT,
    EMBEDDING_MODEL,
)


def load_chunks() -> list[Document]:
    """Load enriched chunks from JSON into LangChain Documents."""
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    docs = []
    for chunk in raw:
        docs.append(
            Document(
                page_content=chunk["page_content"],
                metadata=chunk["metadata"],
            )
        )
    print(f"Loaded {len(docs)} chunks from {CHUNKS_PATH}")
    return docs


def build_vectorstore(docs: list[Document]) -> Chroma:
    """Embed all chunks and persist to ChromaDB."""
    print(f"Embedding {len(docs)} chunks with {EMBEDDING_MODEL}...")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    # Build in batches to show progress
    batch_size = 500
    vectorstore = None

    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                batch,
                embeddings,
                persist_directory=str(CHROMA_DIR),
                collection_name="kmc_docs",
            )
        else:
            vectorstore.add_documents(batch)
        print(
            f"  Embedded {min(i + batch_size, len(docs))}/{len(docs)} chunks",
            flush=True,
        )

    print(f"ChromaDB persisted to {CHROMA_DIR}")
    return vectorstore


def build_bm25(docs: list[Document], k: int = DEFAULT_K) -> BM25Retriever:
    """Build BM25 keyword retriever."""
    print(f"Building BM25 index over {len(docs)} chunks...")
    bm25 = BM25Retriever.from_documents(docs, k=k)
    print("BM25 index ready")
    return bm25


def build_hybrid_retriever(
    vectorstore: Chroma,
    bm25: BM25Retriever,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    k: int = DEFAULT_K,
    fetch_k: int = DEFAULT_FETCH_K,
) -> EnsembleRetriever:
    """Combine vector (MMR) and BM25 into hybrid retrieval with RRF."""
    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k},
    )
    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25],
        weights=[vector_weight, bm25_weight],
    )
    print(
        f"Hybrid retriever ready (vector={vector_weight}, bm25={bm25_weight}, "
        f"k={k}, fetch_k={fetch_k}, search_type=mmr)"
    )
    return ensemble


def sniff_test(retriever: EnsembleRetriever) -> None:
    """Quick sanity check — run a few queries and print results."""
    test_queries = [
        "What was KMC's business model?",
        "Who were the founders?",
        "What technology stack did KMC use?",
    ]

    print("\n=== Sniff Test ===")
    for query in test_queries:
        results = retriever.invoke(query)
        print(f"\nQuery: {query}")
        print(f"  Results: {len(results)}")
        for i, doc in enumerate(results[:3]):
            preview = doc.page_content[:120].replace("\n", " ")
            print(f"  [{i + 1}] {doc.metadata.get('doc_name', '?')} — {preview}...")
    print("\n=== End Sniff Test ===\n")


def build_all() -> EnsembleRetriever:
    """Full Phase 4A pipeline: load chunks → embed → ChromaDB → BM25 → hybrid."""
    docs = load_chunks()
    vectorstore = build_vectorstore(docs)
    bm25 = build_bm25(docs)
    retriever = build_hybrid_retriever(vectorstore, bm25)
    sniff_test(retriever)
    return retriever


if __name__ == "__main__":
    build_all()
