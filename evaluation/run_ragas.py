# ABOUTME: Phase 6 — runs RAGAS evaluation over the RAG agent's answers.
# ABOUTME: Measures faithfulness, answer relevancy, context precision, and context recall.

import json
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

# Suppress deprecation warnings for ragas.metrics imports (v0.4 bug:
# collections metrics don't pass isinstance check in evaluate())
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.config import EMBEDDING_MODEL, EVALUATION_DIR, LLM_MODEL

load_dotenv(Path(__file__).parent.parent / ".env")

# Judge LLM — same OpenRouter model used by the agent
judge_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model=LLM_MODEL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
    )
)

# Embeddings for AnswerRelevancy — reuse the same local model
judge_embeddings = LangchainEmbeddingsWrapper(
    SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
)

# Wire LLM and embeddings into metrics
faithfulness.llm = judge_llm
answer_relevancy.llm = judge_llm
answer_relevancy.embeddings = judge_embeddings
context_precision.llm = judge_llm
context_recall.llm = judge_llm

DATASET_CACHE_PATH = EVALUATION_DIR / "ragas_dataset_cache.json"


def build_ragas_dataset() -> EvaluationDataset:
    """Run each question through retriever + agent, collect responses and contexts."""
    # Check for cached dataset (resumes after crashes)
    if DATASET_CACHE_PATH.exists():
        print(f"Loading cached dataset from {DATASET_CACHE_PATH}...", flush=True)
        with open(DATASET_CACHE_PATH) as f:
            samples = json.load(f)
        print(f"  Loaded {len(samples)} cached samples", flush=True)
        return EvaluationDataset.from_list(samples)

    from src.agent import ask, get_retriever

    ground_truth_path = EVALUATION_DIR / "ground_truth.json"
    with open(ground_truth_path) as f:
        ground_truths = json.load(f)

    retriever = get_retriever()

    samples = []
    for i, gt in enumerate(ground_truths):
        question = gt["question"]
        reference = gt["ground_truth"]

        print(f"  [{i + 1}/{len(ground_truths)}] {question[:70]}...", flush=True)

        # Get retrieved contexts
        retrieved_docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in retrieved_docs]

        # Get agent answer
        response = ask(question)

        samples.append(
            {
                "user_input": question,
                "retrieved_contexts": contexts,
                "response": response,
                "reference": reference,
            }
        )

    # Cache to disk so we don't lose agent responses on crash
    with open(DATASET_CACHE_PATH, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"\n  Dataset cached to {DATASET_CACHE_PATH}", flush=True)

    return EvaluationDataset.from_list(samples)


def run_evaluation() -> None:
    """Build dataset, run RAGAS metrics, save results."""
    output_dir = EVALUATION_DIR / "ragas_results"
    output_dir.mkdir(exist_ok=True)
    scores_cache = output_dir / "ragas_scores.csv"

    # Skip evaluation if scores already exist
    if scores_cache.exists():
        import pandas as pd

        print(f"Loading cached scores from {scores_cache}...", flush=True)
        scores_df = pd.read_csv(scores_cache)
    else:
        print(
            "Building RAGAS dataset (running agent on all questions)...\n", flush=True
        )
        dataset = build_ragas_dataset()

        print(f"\nEvaluating {len(dataset)} samples with RAGAS...\n", flush=True)
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        results = evaluate(dataset=dataset, metrics=metrics)

        scores_df = results.to_pandas()
        scores_df.to_csv(scores_cache, index=False)
        print(f"\nDetailed scores saved to {scores_cache}")

    # Print and save summary
    print("\n=== RAGAS Results ===")
    summary = {}
    for col in scores_df.columns:
        if scores_df[col].dtype in ("float64", "float32", "int64"):
            mean_score = scores_df[col].mean()
            summary[col] = round(mean_score, 3)
            print(f"  {col}: {mean_score:.3f}")

    with open(output_dir / "ragas_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {output_dir / 'ragas_summary.json'}")


if __name__ == "__main__":
    run_evaluation()
