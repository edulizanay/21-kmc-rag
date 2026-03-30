# ABOUTME: Phase 6 — runs RAGAS evaluation over the RAG agent's answers.
# ABOUTME: Measures faithfulness, answer relevancy, context precision, and context recall.

import argparse
import json
import os
import warnings
from datetime import datetime
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


def build_ragas_dataset(question_indices: list[int] | None = None) -> EvaluationDataset:
    """Run each question through retriever + agent, collect responses and contexts.

    If question_indices is provided, only those (0-based) questions are processed
    and the cache is bypassed entirely so fresh agent responses are collected.
    """
    subset_mode = question_indices is not None

    # Full run: load from cache if available
    if not subset_mode and DATASET_CACHE_PATH.exists():
        print(f"Loading cached dataset from {DATASET_CACHE_PATH}...", flush=True)
        with open(DATASET_CACHE_PATH) as f:
            samples = json.load(f)
        print(f"  Loaded {len(samples)} cached samples", flush=True)
        return EvaluationDataset.from_list(samples)

    from src.agent import ask, get_retriever

    ground_truth_path = EVALUATION_DIR / "ground_truth.json"
    with open(ground_truth_path) as f:
        ground_truths = json.load(f)

    if subset_mode:
        invalid = [i for i in question_indices if i >= len(ground_truths)]
        if invalid:
            raise ValueError(
                f"Question indices out of range (max {len(ground_truths) - 1}): {invalid}"
            )
        selected = [(i, ground_truths[i]) for i in question_indices]
        print(
            f"Subset mode: running {len(selected)} question(s) "
            f"(indices {question_indices})\n",
            flush=True,
        )
    else:
        selected = list(enumerate(ground_truths))

    retriever = get_retriever()

    samples = []
    for i, gt in selected:
        question = gt["question"]
        reference = gt["ground_truth"]

        print(f"  [idx {i}] {question[:70]}...", flush=True)

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

    # Always cache agent responses before scoring — safe to re-run scoring if it crashes
    if subset_mode:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subset_cache_path = EVALUATION_DIR / f"ragas_dataset_cache_subset_{timestamp}.json"
        with open(subset_cache_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"\n  Agent responses cached to {subset_cache_path}", flush=True)
    else:
        with open(DATASET_CACHE_PATH, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"\n  Dataset cached to {DATASET_CACHE_PATH}", flush=True)

    return EvaluationDataset.from_list(samples)


def run_evaluation(question_indices: list[int] | None = None) -> None:
    """Build dataset, run RAGAS metrics, save results."""
    output_dir = EVALUATION_DIR / "ragas_results"
    output_dir.mkdir(exist_ok=True)

    subset_mode = question_indices is not None

    if subset_mode:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scores_path = output_dir / f"ragas_scores_subset_{timestamp}.csv"
    else:
        scores_path = output_dir / "ragas_scores.csv"

    # Full run: skip if scores already exist
    if not subset_mode and scores_path.exists():
        import pandas as pd

        print(f"Loading cached scores from {scores_path}...", flush=True)
        scores_df = pd.read_csv(scores_path)
    else:
        print(
            f"Building RAGAS dataset (running agent on "
            f"{'selected' if subset_mode else 'all'} questions)...\n",
            flush=True,
        )
        dataset = build_ragas_dataset(question_indices)

        print(f"\nEvaluating {len(dataset)} samples with RAGAS...\n", flush=True)
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        from ragas import RunConfig

        # context_precision makes ~k sequential LLM calls per question (~34s each).
        # timeout=300 gives 8 × 34s = 272s headroom; default 120s always fails.
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            run_config=RunConfig(timeout=300),
        )

        scores_df = results.to_pandas()
        scores_df.to_csv(scores_path, index=False)
        print(f"\nDetailed scores saved to {scores_path}")

    # Print and save summary
    print("\n=== RAGAS Results ===")
    import math

    summary = {}
    for col in scores_df.columns:
        if scores_df[col].dtype in ("float64", "float32", "int64"):
            mean_score = scores_df[col].mean()
            # Store null for all-NaN metrics (e.g. context_precision timing out)
            summary[col] = None if math.isnan(mean_score) else round(mean_score, 3)
            print(f"  {col}: {mean_score:.3f}")

    if not subset_mode:
        with open(output_dir / "ragas_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {output_dir / 'ragas_summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the RAG agent."
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help=(
            "Comma-separated 0-based question indices to evaluate (e.g. '0,3,9'). "
            "Bypasses cache and writes to a timestamped subset CSV. "
            "Omit to run the full evaluation."
        ),
    )
    args = parser.parse_args()

    indices = (
        [int(x.strip()) for x in args.questions.split(",")]
        if args.questions
        else None
    )
    run_evaluation(indices)
