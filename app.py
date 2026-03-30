# ABOUTME: Streamlit frontend for the KMC RAG chatbot.
# ABOUTME: Two tabs: Chat (hiring managers ask questions) and Evaluation (retrieval benchmarks).

import json
from pathlib import Path

import streamlit as st

from src.agent import ask_with_sources
from src.call_cap import check_and_increment

# --- Page config ---

st.set_page_config(
    page_title="KMC — Ask Eduardo's Experience",
    page_icon="💬",
    layout="centered",
)

# --- Constants ---

STARTER_PROMPTS = [
    "What was Eduardo's role at KMC?",
    "How did KMC grow under his leadership?",
    "What was KMC's regulatory strategy?",
]

RETRIEVAL_RESULTS_PATH = Path("evaluation/retrieval_test_results.json")
RAGAS_RESULTS_DIR = Path("evaluation/ragas_results")

# --- Tabs ---

chat_tab, eval_tab = st.tabs(["Chat", "Evaluation"])

# ── Tab 1: Chat ────────────────────────────────────────────────────────────────

with chat_tab:
    st.title("Ask about Eduardo's experience at KMC")
    st.caption(
        "KMC (KeepMeCompany) was an AI-powered healthcare communications startup. "
        "Ask anything about the company, its products, team, or strategy."
    )

    # Initialise session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"- {src}")

    # Starter prompts — shown only before the first message
    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        cols = st.columns(len(STARTER_PROMPTS))
        for col, prompt in zip(cols, STARTER_PROMPTS):
            if col.button(prompt, use_container_width=True):
                st.session_state._pending_prompt = prompt
                st.rerun()

    # Handle a starter prompt click (set in session_state above)
    if "_pending_prompt" in st.session_state:
        user_input = st.session_state.pop("_pending_prompt")
    else:
        user_input = st.chat_input("Ask a question...")

    if user_input:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Check daily cap
        if not check_and_increment():
            reply = {
                "role": "assistant",
                "content": (
                    "The daily question limit has been reached. "
                    "Please check back tomorrow or contact Eduardo directly."
                ),
                "sources": [],
            }
            st.session_state.messages.append(reply)
            with st.chat_message("assistant"):
                st.markdown(reply["content"])
        else:
            # Call the agent
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = ask_with_sources(user_input)

                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("Sources"):
                        for src in result["sources"]:
                            st.markdown(f"- {src}")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                }
            )

# ── Tab 2: Evaluation ──────────────────────────────────────────────────────────

with eval_tab:
    st.title("Evaluation results")

    # ── Retrieval benchmark ──

    st.subheader("Retrieval benchmark")

    if RETRIEVAL_RESULTS_PATH.exists():
        with open(RETRIEVAL_RESULTS_PATH) as f:
            results = json.load(f)

        passed = sum(1 for r in results if r["status"] == "PASS")
        total = len(results)
        accuracy = passed / total * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.0f}%")
        col2.metric("Passed", passed)
        col3.metric("Failed", total - passed)

        rows = [
            {
                "Question": r["question"],
                "Difficulty": r["difficulty"],
                "Status": r["status"],
                "Hits": ", ".join(r.get("hits", [])),
                "Misses": ", ".join(r.get("misses", [])),
            }
            for r in results
        ]
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("Retrieval test results not found.")

    st.divider()

    # ── RAGAS metrics ──

    st.subheader("RAGAS metrics")

    ragas_files = list(RAGAS_RESULTS_DIR.glob("*.json")) + list(
        RAGAS_RESULTS_DIR.glob("*.csv")
    )

    if ragas_files:
        # Load the most recent file
        latest = sorted(ragas_files)[-1]
        if latest.suffix == ".json":
            with open(latest) as f:
                ragas_data = json.load(f)
            metric_keys = [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ]
            cols = st.columns(len(metric_keys))
            for col, key in zip(cols, metric_keys):
                value = ragas_data.get(key)
                if value is not None:
                    col.metric(key.replace("_", " ").title(), f"{float(value):.2f}")
        else:
            import pandas as pd

            df = pd.read_csv(latest)
            st.dataframe(df, use_container_width=True)
    else:
        st.info(
            "RAGAS evaluation has not been run yet (Phase 6). Results will appear here automatically once available."
        )
