# ABOUTME: Phase 5 — LangGraph agent with router, RAG search, doc specialist, and "I don't know" tools.
# ABOUTME: Entry point for the RAG chatbot — takes a question and returns a grounded answer.

import csv
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from src.config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    INVENTORY_PATH,
    LLM_MODEL,
    UNANSWERED_LOG_PATH,
)
from src.vectorstore import load_chunks

load_dotenv(Path(__file__).parent.parent / ".env")

# --- LLM setup (OpenRouter, OpenAI-compatible) ---

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file or Streamlit Cloud secrets."
            )
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.1,
        )
    return _llm


# --- Retriever (lazy-loaded on first use) ---

_retriever = None


def get_retriever():
    """Build hybrid retriever on first call, then cache."""
    global _retriever
    if _retriever is not None:
        return _retriever

    from langchain_chroma import Chroma
    from langchain_community.embeddings import SentenceTransformerEmbeddings

    from src.vectorstore import build_bm25, build_hybrid_retriever, build_vectorstore

    docs = load_chunks()
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    # Rebuild vectorstore from chunks if chroma_db doesn't exist (e.g. cloud deploy)
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        print("ChromaDB not found — rebuilding from chunks.json...", flush=True)
        vectorstore = build_vectorstore(docs)
    else:
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_name="kmc_docs",
        )

    bm25 = build_bm25(docs)
    _retriever = build_hybrid_retriever(vectorstore, bm25)
    return _retriever


# --- Tools ---


@tool
def rag_search(query: str) -> str:
    """Search KeepMeCompany's document corpus for information relevant to the query.
    Use this for any question about the company, its products, team, strategy, or operations."""
    retriever = get_retriever()
    results = retriever.invoke(query)
    if not results:
        return (
            "No relevant documents found for this query. "
            "Unless you have already retrieved relevant context from a previous search in this session, "
            "you should call i_dont_know."
        )

    chunks = []
    for doc in results:
        source = doc.metadata.get("doc_name", "unknown")
        chunks.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(chunks)


@tool
def doc_specialist(doc_name: str, question: str) -> str:
    """Load a full document and answer a specific question about it.
    Use when RAG chunks are ambiguous or you need deeper detail from a specific document.
    doc_name: the filename to look up (e.g., 'Series A Strategy.pdf')
    question: the specific question to answer about this document."""
    from src.config import PROCESSED_TEXTS_PATH

    # Look up doc_id from inventory
    with open(INVENTORY_PATH, encoding="utf-8") as f:
        rows = {r["filename"]: r for r in csv.DictReader(f)}

    if doc_name not in rows:
        return f"Document '{doc_name}' not found in inventory."

    doc_id = rows[doc_name]["doc_id"]

    # Load pre-extracted text (works both locally and on Streamlit Cloud)
    with open(PROCESSED_TEXTS_PATH) as f:
        processed = json.load(f)

    if doc_id not in processed:
        return f"No processed text found for '{doc_name}' ({doc_id})."

    text = processed[doc_id]["text"]
    if not text.strip():
        return f"Could not extract text from '{doc_name}'."

    # Ask LLM to answer the question grounded in the document
    prompt = (
        f"Answer the following question based ONLY on this document. "
        f"If the document doesn't contain the answer, say so.\n\n"
        f"Document: {doc_name}\n\n"
        f"Content:\n{text[:30_000]}\n\n"
        f"Question: {question}"
    )
    response = _get_llm().invoke(prompt)
    return response.content


@tool
def i_dont_know(question: str) -> str:
    """Use when the question cannot be answered from KeepMeCompany's documents.
    Logs the unanswered question for later review."""
    entry = {
        "question": question,
        "timestamp": datetime.now().isoformat(),
    }

    log = []
    if UNANSWERED_LOG_PATH.exists():
        with open(UNANSWERED_LOG_PATH) as f:
            log = json.load(f)

    log.append(entry)
    with open(UNANSWERED_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    return (
        "I don't have enough information in KeepMeCompany's documents to answer that question. "
        "I've logged it for Eduardo to review."
    )


# --- Graph ---

tools = [rag_search, doc_specialist, i_dont_know]
tools_by_name = {t.name: t for t in tools}
_llm_with_tools = None


def _get_llm_with_tools():
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = _get_llm().bind_tools(tools)
    return _llm_with_tools


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about KeepMeCompany, "
    "an AI-powered healthcare communications startup founded by Eduardo Lizana and Rodrigo Orpis. "
    "Use your tools to search the company's document corpus before answering. "
    "Always ground your answers in the retrieved documents. "
    "If the retrieved documents are clearly unrelated to the question, or if rag_search returns "
    "no results and you have no prior context from an earlier search in this session, "
    "you MUST call i_dont_know — do not answer from general knowledge. "
    "For questions that require combining facts from multiple sources "
    "(e.g. questions about both X and Y, or questions comparing two documents), "
    "call rag_search multiple times with different focused queries, then synthesize the results. "
    "Be concise and cite which document your answer comes from."
)


def router(state: MessagesState):
    """Call the LLM to decide which tool to use."""
    response = _get_llm_with_tools().invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
    return {"messages": [response]}


def tool_node(state: MessagesState):
    """Execute the tool calls from the LLM response."""
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_fn = tools_by_name[tool_call["name"]]
        observation = tool_fn.invoke(tool_call["args"])
        results.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )
    return {"messages": results}


def should_continue(state: MessagesState):
    """Route to tool_node if there are tool calls, otherwise end."""
    last = state["messages"][-1]
    if last.tool_calls:
        return "tool_node"
    return END


# Build the graph
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("router", router)
graph_builder.add_node("tool_node", tool_node)
graph_builder.add_edge(START, "router")
graph_builder.add_conditional_edges("router", should_continue, ["tool_node", END])
graph_builder.add_edge("tool_node", "router")

agent = graph_builder.compile()


def ask(question: str) -> str:
    """Ask the agent a question and return the final answer."""
    result = agent.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content


def ask_with_sources(question: str, chat_history: list | None = None) -> dict:
    """Ask the agent a question and return the answer with source documents.

    Returns {"answer": str, "sources": list[str]} where sources are
    extracted from the rag_search tool's output in the message history.

    chat_history: list of {"role": "user"|"assistant", "content": str} dicts
    from prior turns, used to give the agent conversational context.
    """
    import re

    # Keep only the last 10 exchanges to avoid exceeding the LLM context window
    MAX_HISTORY_TURNS = 10
    recent_history = (chat_history or [])[-MAX_HISTORY_TURNS * 2 :]

    messages = []
    for msg in recent_history:
        messages.append((msg["role"], msg["content"]))
    messages.append(("user", question))

    result = agent.invoke({"messages": messages})
    answer = result["messages"][-1].content

    # Extract sources from tool messages (rag_search embeds [Source: name] tags)
    sources = []
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            sources.extend(re.findall(r"\[Source:\s*([^\]]+)\]", msg.content))

    return {"answer": answer, "sources": list(dict.fromkeys(sources))}


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What did KeepMeCompany do?"
    print(ask(question))
