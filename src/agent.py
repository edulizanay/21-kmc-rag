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

llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.1,
)

# --- Retriever (lazy-loaded on first use) ---

_retriever = None


def get_retriever():
    """Build hybrid retriever on first call, then cache."""
    global _retriever
    if _retriever is not None:
        return _retriever

    from langchain_chroma import Chroma
    from langchain_community.embeddings import SentenceTransformerEmbeddings

    from src.vectorstore import build_bm25, build_hybrid_retriever

    docs = load_chunks()
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
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
        return "No relevant documents found."

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
    # Find the document path from inventory
    with open(INVENTORY_PATH, encoding="utf-8") as f:
        rows = {r["filename"]: r for r in csv.DictReader(f)}

    if doc_name not in rows:
        return f"Document '{doc_name}' not found in inventory."

    filepath = rows[doc_name]["path"]
    if not Path(filepath).exists():
        return f"Document file not found at {filepath}."

    # Extract full text
    from src.extract_text import extract_text

    text = extract_text(filepath, max_chars=50_000)
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
    response = llm.invoke(prompt)
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
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about KeepMeCompany, "
    "an AI-powered healthcare communications startup founded by Eduardo Lizana and Rodrigo Orpis. "
    "Use your tools to search the company's document corpus before answering. "
    "Always ground your answers in the retrieved documents. "
    "If the documents don't contain enough information, use the i_dont_know tool. "
    "Be concise and cite which document your answer comes from."
)


def router(state: MessagesState):
    """Call the LLM to decide which tool to use."""
    response = llm_with_tools.invoke(
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


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What did KeepMeCompany do?"
    print(ask(question))
