# ABOUTME: Streamlit frontend for the KMC RAG chatbot.
# ABOUTME: Three tabs: Chat, Architecture (Graphviz diagrams), and Evaluation (retrieval benchmarks).

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

# Inject Streamlit Cloud secrets into os.environ so downstream code can use os.getenv()
try:
    os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
except BaseException:
    pass

import src.agent as _agent_mod
from src.agent import get_retriever, stream_with_sources
from src.call_cap import check_and_increment

# --- Page config ---

st.set_page_config(
    page_title="KMC — Ask Eduardo's Experience",
    page_icon="💬",
    layout="centered",
)


# --- Eager retriever init (cached across all sessions) ---


@st.cache_resource(show_spinner=False)
def _warm_retriever():
    """Load the hybrid retriever once and cache across all Streamlit sessions."""
    return get_retriever()


_init_placeholder = st.empty()
if not st.session_state.get("_retriever_ready"):
    with _init_placeholder.status("Loading knowledge base...", expanded=True) as s:
        st.write("Loading vector store and building search index...")
        _warm_retriever()
        s.update(label="Knowledge base ready", state="complete", expanded=False)
    st.session_state._retriever_ready = True
    _init_placeholder.empty()
else:
    _warm_retriever()

# Re-set the module global on every script rerun so it survives module reimports
_agent_mod._retriever = _warm_retriever()

# --- Constants ---

STARTER_PROMPTS = [
    "What was Eduardo's role at KMC?",
    "How did KMC grow under his leadership?",
    "What was KMC's regulatory strategy?",
]

RETRIEVAL_RESULTS_PATH = Path("evaluation/retrieval_test_results.json")
RAGAS_RESULTS_DIR = Path("evaluation/ragas_results")

# --- Tabs ---

chat_tab, eval_tab, arch_tab = st.tabs(["Chat", "Evaluation", "Architecture"])

# ── Tab 1: Chat ────────────────────────────────────────────────────────────────

with chat_tab:
    st.title("Ask about KeepmeCompany")
    st.caption(
        "KMC (KeepMeCompany) was an AI-powered healthcare communications startup founded by Rodrigo Orpis and Eduardo Lizana. "
        "Ask anything about the company, its products, team, or strategy."
    )

    # Initialise session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Scrollable container for chat history — input bar stays below
    chat_container = st.container(height=500)

    with chat_container:
        # Render chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander("Sources"):
                        for src in msg["sources"]:
                            st.markdown(f"- {src}")

        # Starter prompts — shown only before the first message
        if not st.session_state.messages and "_pending_prompt" not in st.session_state:
            st.markdown("**Try asking:**")
            cols = st.columns(len(STARTER_PROMPTS))
            for col, prompt in zip(cols, STARTER_PROMPTS):
                if col.button(prompt, use_container_width=True):
                    st.session_state._pending_prompt = prompt
                    st.rerun()

    # Chat input is always rendered below the container
    typed_input = st.chat_input("Ask a question...")

    # Handle a starter prompt click (set in session_state above)
    if "_pending_prompt" in st.session_state:
        user_input = st.session_state.pop("_pending_prompt")
    else:
        user_input = typed_input

    if user_input:
        # Show user message inside the scrollable container
        st.session_state.messages.append({"role": "user", "content": user_input})
        with chat_container:
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
                    placeholder = st.empty()
                    placeholder.markdown("*Thinking...*")

                    chat_history = st.session_state.messages[:-1]
                    result = None

                    import time

                    for event in stream_with_sources(user_input, chat_history):
                        if event["type"] == "sources":
                            placeholder.markdown(f"*{event['sources'][0]}*")
                            time.sleep(
                                0.1
                            )  # yield to Tornado event loop to flush websocket update
                        elif event["type"] == "answer":
                            result = event

                    placeholder.empty()

                    answer = (
                        result["answer"] if result else "Sorry, something went wrong."
                    )
                    answer_sources = result["sources"] if result else []

                    st.markdown(answer)
                    if answer_sources:
                        with st.expander("Sources"):
                            for src in answer_sources:
                                st.markdown(f"- {src}")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": answer_sources,
                        }
                    )

# ── Tab 2: Architecture ────────────────────────────────────────────────────────

QUERY_FLOW_GRAPH = """
digraph query_flow {
    rankdir=TD
    node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11]
    edge [fontname="Helvetica" fontsize=10]

    Q [label=<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><B>Your Question</B></TD></TR>
        </TABLE>
    > fillcolor="#e8f4f8"]

    subgraph cluster_agent {
        label=<<B>LangGraph Agent</B>>
        style="dashed"
        color="#666666"
        fontname="Helvetica"

        Router [label=<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
            <TR><TD><B>Router</B></TD></TR>
            <TR><TD><FONT POINT-SIZE="9">Decides the best strategy</FONT></TD></TR>
            </TABLE>
        > fillcolor="#d4e6f1"]

        RAG [label=<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
            <TR><TD><B>RAG Search</B></TD></TR>
            <TR><TD><FONT POINT-SIZE="9">Vector + BM25 hybrid, MMR diversity</FONT></TD></TR>
            </TABLE>
        > fillcolor="#d5f5e3"]

        Doc [label=<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
            <TR><TD><B>Doc Specialist</B></TD></TR>
            <TR><TD><FONT POINT-SIZE="9">Loads full document for spreadsheets</FONT></TD></TR>
            <TR><TD><FONT POINT-SIZE="9">and large structured files</FONT></TD></TR>
            </TABLE>
        > fillcolor="#d5f5e3"]

        IDK [label=<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
            <TR><TD><B>"I don't know" tool</B></TD></TR>
            <TR><TD><FONT POINT-SIZE="9">Logs to unanswered_log.json</FONT></TD></TR>
            </TABLE>
        > fillcolor="#fadbd8"]

        Eval [label=<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
            <TR><TD><B>Evaluate Quality</B></TD></TR>
            </TABLE>
        > fillcolor="#fdebd0"]
    }

    Ans [label=<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><B>Final Answer</B></TD></TR>
        <TR><TD><FONT POINT-SIZE="9">with source citations</FONT></TD></TR>
        </TABLE>
    > fillcolor="#e8f4f8"]

    Q -> Router
    Router -> RAG [label="chunk-based"]
    Router -> Doc [label="full-doc needed"]
    Router -> IDK [label="not in corpus"]
    RAG -> Eval
    Doc -> Eval
    Eval -> Ans [label="good enough"]
    Eval -> Router [label="retry" style=dashed]
}
"""

DATA_PIPELINE_GRAPH = """
digraph pipeline {
    rankdir=TD
    node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11]
    edge [fontname="Helvetica" fontsize=10]

    Raw [label=<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><B>Raw Files</B></TD></TR>
        <TR><TD><FONT POINT-SIZE="9">503 docs — PDF, DOCX, XLSX, PPTX, MD, CSV, TXT</FONT></TD></TR>
        </TABLE>
    > fillcolor="#e8f4f8"]

    P1 [label=<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><B>Phase 1: Inventory &amp; Triage</B></TD></TR>
        <TR><TD><FONT POINT-SIZE="9">LLM classifies each file as include/exclude</FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="9">351 included, 152 excluded</FONT></TD></TR>
        </TABLE>
    > fillcolor="#d5f5e3"]

    P2 [label=<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><B>Phase 2: Text Extraction</B></TD></TR>
        <TR><TD><FONT POINT-SIZE="9">Format-specific parsers (PDF, DOCX, XLSX...)</FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="9">Clean headers, footers, page artifacts</FONT></TD></TR>
        </TABLE>
    > fillcolor="#d5f5e3"]

    P3 [label=<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><B>Phase 3: Chunking</B></TD></TR>
        <TR><TD><FONT POINT-SIZE="9">512 tokens, 50 overlap</FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="9">Markdown files split by headers</FONT></TD></TR>
        </TABLE>
    > fillcolor="#d5f5e3"]

    P3b [label=<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><B>Phase 3b: Contextual Enrichment</B></TD></TR>
        <TR><TD><FONT POINT-SIZE="9">Claude Haiku generates a per-chunk</FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="9">content prefix for retrieval</FONT></TD></TR>
        </TABLE>
    > fillcolor="#fdebd0"]

    P4 [label=<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><B>Phase 4: Embedding</B></TD></TR>
        <TR><TD><FONT POINT-SIZE="9">all-MiniLM-L6-v2 (384 dims, local)</FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="9">Stored in ChromaDB</FONT></TD></TR>
        </TABLE>
    > fillcolor="#d5f5e3"]

    subgraph cluster_hybrid {
        label=<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
            <TR><TD><B>Hybrid Retriever</B></TD></TR>
            <TR><TD><FONT POINT-SIZE="9">Reciprocal Rank Fusion</FONT></TD></TR>
            </TABLE>
        >
        style="dashed"
        color="#666666"
        fontname="Helvetica"

        Vec [label=<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
            <TR><TD><B>Vector Search (ChromaDB)</B></TD></TR>
            <TR><TD><FONT POINT-SIZE="9">Semantic similarity — weight 0.6</FONT></TD></TR>
            </TABLE>
        > fillcolor="#d4e6f1"]

        BM25 [label=<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
            <TR><TD><B>BM25 Keyword Search</B></TD></TR>
            <TR><TD><FONT POINT-SIZE="9">Exact term matching — weight 0.4</FONT></TD></TR>
            </TABLE>
        > fillcolor="#d4e6f1"]
    }

    Result [label=<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><B>11,710 enriched chunks</B></TD></TR>
        <TR><TD><FONT POINT-SIZE="9">ready for retrieval</FONT></TD></TR>
        </TABLE>
    > fillcolor="#e8f4f8"]

    # Annotation nodes — data examples at key stages
    note_p2 [shape=box style="rounded,filled" label=<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">
        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9"><B>e.g.</B> Pilot Proposal.pptx</FONT></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#555555">[Slide 3] Objectives of the pilot —</FONT></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#555555">Measure and compare expected vs.</FONT></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#555555">actual time savings for clinicians</FONT></TD></TR>
        </TABLE>
    > fillcolor="#fffde7"]

    note_p3b [shape=box style="rounded,filled" label=<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">
        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9"><B>Without enrichment:</B></FONT></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#555555">"A - Asthma: Do you have asthma?</FONT></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#555555"> I - Inhaler: Have you used it?"</FONT></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9"><B>With Haiku prefix:</B></FONT></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#555555">"AIR protocol for breathing difficulty</FONT></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#555555"> with emergency triggers for cyanosis:</FONT></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#555555"> A - Asthma: Do you have asthma? ..."</FONT></TD></TR>
        </TABLE>
    > fillcolor="#fffde7"]

    note_hybrid [shape=box style="rounded,filled" label=<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">
        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9"><B>Query: "What did seniors experience?"</B></FONT></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#2e7d32">Vector  → finds "elderly users reported"</FONT></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#c62828">BM25   → misses (no keyword match)</FONT></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9"><B>Query: "Severin Hoegl"</B></FONT></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#c62828">Vector  → misses (rare proper noun)</FONT></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="8" COLOR="#2e7d32">BM25   → finds exact name match</FONT></TD></TR>
        </TABLE>
    > fillcolor="#fffde7"]

    # Connect annotations with dotted lines
    note_p2 -> P2 [style=dotted arrowhead=none color="#cccccc"]
    note_p3b -> P3b [style=dotted arrowhead=none color="#cccccc"]
    note_hybrid -> Result [style=dotted arrowhead=none color="#cccccc"]

    # Main flow
    Raw -> P1 -> P2 -> P3 -> P3b -> P4
    P4 -> Vec
    P4 -> BM25
    Vec -> Result
    BM25 -> Result

    # Push annotations to the right
    {rank=same; P2; note_p2}
    {rank=same; P3b; note_p3b}
    {rank=same; Result; note_hybrid}
}
"""


with arch_tab:
    st.title("System architecture")
    st.caption("How the RAG pipeline processes documents and answers questions.")

    # ── Query flow ──
    st.subheader("Query flow")
    st.markdown(
        "When you ask a question, a **LangGraph router agent** decides the best "
        "strategy: search over chunks, load a full document, or decline gracefully. "
        "A quality evaluator checks the result before returning it."
    )
    st.graphviz_chart(QUERY_FLOW_GRAPH)

    st.divider()

    # ── Data pipeline ──
    st.subheader("Data pipeline")
    st.markdown(
        "Raw files go through four processing phases before they're searchable. "
        "Each chunk gets an LLM-generated prefix describing its content, then "
        "gets indexed into a **hybrid retriever** that combines semantic and keyword search."
    )
    st.graphviz_chart(DATA_PIPELINE_GRAPH)

    st.divider()

    # ── Key stats ──
    st.subheader("Corpus stats")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files cataloged", "503")
    c2.metric("Docs indexed", "351")
    c3.metric("Chunks", "11,710")
    c4.metric("Embedding dims", "384")

# ── Tab 3: Evaluation ──────────────────────────────────────────────────────────

with eval_tab:
    st.title("Evaluation results")

    # ── Retrieval benchmark ──

    st.subheader("Retrieval benchmark")

    if RETRIEVAL_RESULTS_PATH.exists():
        with open(RETRIEVAL_RESULTS_PATH) as f:
            results = json.load(f)

        answerable = [r for r in results if r["difficulty"] != "unanswerable"]
        passed = sum(1 for r in answerable if r["status"] == "PASS")
        failed = sum(1 for r in answerable if r["status"] == "FAIL")
        total = len(answerable)
        accuracy = passed / total * 100 if total else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.0f}%")
        col2.metric("Passed", passed)
        col3.metric("Failed", failed)

        rows = [
            {
                "Question": f"{i + 1}. {r['question']}",
                "Status": r["status"],
            }
            for i, r in enumerate(answerable)
        ]
        df = pd.DataFrame(rows)
        styled = df.style.map(
            lambda v: "color: #4caf50" if v == "PASS" else "color: #e53935",
            subset=["Status"],
        )
        st.dataframe(
            styled,
            use_container_width=True,
            column_config={
                "Question": st.column_config.Column(width=600),
                "Status": st.column_config.Column(width=60),
            },
            hide_index=True,
        )
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
            phase6 = {
                "faithfulness": 0.587,
                "answer_relevancy": 0.844,
                "context_precision": 0.278,
                "context_recall": 0.300,
            }
            cols = st.columns(len(metric_keys))
            for col, key in zip(cols, metric_keys):
                value = ragas_data.get(key)
                label = key.replace("_", " ").title()
                if value is None:
                    col.metric(label, "N/A")
                else:
                    baseline = phase6.get(key)
                    delta = (
                        round(float(value) - baseline, 3)
                        if baseline is not None
                        else None
                    )
                    delta_str = (
                        f"{delta:+.3f} vs Phase 6" if delta is not None else None
                    )
                    col.metric(label, f"{float(value):.3f}", delta=delta_str)

            with st.expander("What do these metrics mean?"):
                st.markdown(
                    "All metrics are scored **0 to 1** (higher is better).\n\n"
                    "| Metric | What it measures |\n"
                    "|---|---|\n"
                    "| **Faithfulness** | How factually consistent the answer is with the retrieved context. "
                    'e.g. if the docs say "founded in 2020" but the answer says 2019, faithfulness drops. |\n'
                    "| **Answer Relevancy** | How relevant the answer is to the question asked. "
                    "e.g. asking about revenue but getting an answer about the tech stack scores low. |\n"
                    "| **Context Precision** | Whether the most relevant chunks are ranked highest in retrieval. "
                    "e.g. asking about founders but the top results are about marketing scores low. |\n"
                    "| **Context Recall** | How much of the ground-truth answer is covered by retrieved context. "
                    "e.g. if the answer needs 3 key facts but retrieval only finds 1 of them, recall is low. |\n\n"
                    "---\n\n"
                    "**Example: what RAGAS sees for one question**\n\n"
                    "*Input to RAGAS:*\n\n"
                    "```\n"
                    "question:          'What is the annual revenue of Acme Corp?'\n"
                    "answer:            'Acme Corp reported annual revenue of $4.2M in 2023.'\n"
                    "retrieved_context: [\n"
                    "  [rank 1] 'Acme Corp 2023 annual report: total revenue $4.2M, up 18% YoY.',  ✓ relevant\n"
                    "  [rank 2] 'Acme Corp was founded in 2018 and is headquartered in Madrid.',    ✗ not relevant\n"
                    "  [rank 3] 'Acme Corp revenue breakdown: 60% SaaS, 40% services in 2023.',     ✓ relevant\n"
                    "  [rank 4] 'Acme Corp hired 50 engineers in Q3 2023.',                         ✗ not relevant\n"
                    "  [rank 5] 'Acme Corp CEO joined from Google in 2021.',                        ✗ not relevant\n"
                    "]\n"
                    "ground_truth:      'Acme Corp had annual revenue of $4.2M in 2023.'\n"
                    "```\n\n"
                    "*Output from RAGAS:*\n\n"
                    "```\n"
                    "faithfulness:      1.00  ← answer is fully supported by the retrieved chunks\n"
                    "answer_relevancy:  0.95  ← LLM reverse-generates questions from the answer and measures\n"
                    "                          cosine similarity to the original; rarely perfect even for\n"
                    "                          direct answers due to the round-trip through the LLM\n"
                    "context_precision: 0.83  ← relevant chunks are at ranks 1 and 3\n"
                    "                          precision@1 = 1/1 = 1.00 (rank 1 is relevant)\n"
                    "                          precision@3 = 2/3 = 0.67 (2 relevant in top 3)\n"
                    "                          average over relevant positions = (1.00 + 0.67) / 2 = 0.83\n"
                    "context_recall:    1.00  ← retrieved context covers everything in the ground truth\n"
                    "```"
                )
        else:
            import pandas as pd

            df = pd.read_csv(latest)
            st.dataframe(df, use_container_width=True)
    else:
        st.info(
            "RAGAS evaluation has not been run yet (Phase 6). Results will appear here automatically once available."
        )
