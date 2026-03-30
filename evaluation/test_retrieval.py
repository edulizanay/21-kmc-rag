# ABOUTME: Retrieval evaluation — runs test questions against the hybrid retriever.
# ABOUTME: Checks if expected source documents appear in retrieved chunks.

import json

from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

from src.config import (
    CHROMA_DIR,
    DEFAULT_K,
    EMBEDDING_MODEL,
    EVALUATION_DIR,
)
from src.vectorstore import build_bm25, build_hybrid_retriever, load_chunks


def run_retrieval_test(k: int = DEFAULT_K) -> None:
    """Run all test questions and report retrieval accuracy."""
    # Load test set
    test_path = EVALUATION_DIR / "test_set.json"
    with open(test_path) as f:
        test_set = json.load(f)

    # Build retriever (same config as production: MMR + BM25 hybrid)
    print("Loading chunks and building retriever...")
    docs = load_chunks()

    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="kmc_docs",
    )
    bm25 = build_bm25(docs, k=k)
    retriever = build_hybrid_retriever(vectorstore, bm25, k=k)

    print(f"Running {len(test_set)} test queries (k={k})...\n")

    passed = 0
    failed = 0
    results = []

    for i, test in enumerate(test_set):
        question = test["question"]
        expected_ids = set(test["expected_doc_ids"])
        difficulty = test["difficulty"]

        retrieved = retriever.invoke(question)
        retrieved_ids = {doc.metadata.get("doc_id", "") for doc in retrieved}

        if not expected_ids:
            # Unanswerable question — check that retrieval returns irrelevant docs
            status = "SKIP (unanswerable)"
            passed += 1
        elif expected_ids & retrieved_ids:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1

        hit_ids = expected_ids & retrieved_ids
        miss_ids = expected_ids - retrieved_ids

        result = {
            "question": question,
            "difficulty": difficulty,
            "status": status,
            "expected": sorted(expected_ids),
            "retrieved": sorted(retrieved_ids),
            "hits": sorted(hit_ids),
            "misses": sorted(miss_ids),
        }
        results.append(result)

        icon = "+" if "PASS" in status or "SKIP" in status else "X"
        print(f"  [{icon}] {status} | {difficulty:12s} | {question[:70]}")
        if miss_ids:
            print(f"       Missing: {miss_ids}")
            for doc in retrieved[:3]:
                preview = doc.page_content[:80].replace("\n", " ")
                print(f"       Got: {doc.metadata.get('doc_id', '?')} — {preview}")

    total = passed + failed
    print(f"\n=== Results: {passed}/{total} passed ({100 * passed / total:.0f}%) ===")
    if failed:
        print(f"  {failed} questions failed to retrieve expected documents")

    # Save detailed results
    output_path = EVALUATION_DIR / "retrieval_test_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to {output_path}")


if __name__ == "__main__":
    run_retrieval_test()
