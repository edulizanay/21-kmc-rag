# KMC RAG Portfolio Project — Development Plan v3

## Project summary

A RAG-powered chatbot over KeepmeCompany's document corpus (~150-200 docs, mixed formats) that lets hiring managers explore Eduardo's startup experience conversationally. The system demonstrates end-to-end RAG + agentic orchestration using LangGraph.

---

## Rules

These apply across all phases:

1. **No Anthropic API calls from code.** For LLM work during build (metadata, triage, enrichment), spawn Claude Code Haiku subagents. For runtime LLM calls (agent, doc specialist), use OpenRouter API with `qwen/qwen3.5-flash-02-23`.
2. **Use Haiku subagents for bulk doc processing.** Spawn 3 at a time in parallel. Each processes a batch and marks files as completed in the inventory.
3. **Keep code simple.** Structure matters more than quality at this stage. Short prompts — Edu will refine them later.
4. **Don't duplicate source files.** Reference originals in place. Only processed/chunked versions live in our project.
5. **Ignore video files.** Skip `.mp4`, `.mov`, `.avi`, and similar.
6. **Log architectural decisions.** Every meaningful tradeoff gets a concise entry in `architectural_decisions.md`.
7. **File count verification.** At every phase, inventory count must match actual folder count (minus excluded files).
8. **Preserve main agent context.** The main agent should never read document contents directly. Spawn Haiku subagents for any work that requires reading file contents (triage, metadata, enrichment). This keeps the main agent's context window clean for orchestration.

---

## Architecture (v1)

### High-level flow

```
Question → Router Agent → [RAG Search | Doc Specialist | "I don't know"] → Evaluate Quality → [Retry | Generate Answer]
```

**Nodes:**
- **Router**: receives query, decides which tool to call
- **RAG Search**: hybrid retrieval (vector + BM25) over chunked corpus
- **Doc Specialist**: loads a full document, answers a specific question grounded in it. Called when chunks are ambiguous, conflicting, or sparse.
- **"I don't know"**: logs unanswered questions to `unanswered_log.json`, returns graceful message
- **Evaluate**: checks retrieval quality, decides to retry (loop back to router) or generate final answer

### Data stores

- **Vector store** (ChromaDB): all docs chunked with contextual metadata
- **Full documents store**: original files at source path, for the doc specialist
- **Unanswered log**: `unanswered_log.json` in project root

### Tech stack

| Component | Tool | Cost |
|-----------|------|------|
| Orchestration | LangGraph (MIT) | Free |
| Building blocks | LangChain core + community | Free |
| Vector store | ChromaDB (local) | Free |
| Keyword search | BM25Retriever (rank_bm25) | Free |
| Hybrid search | EnsembleRetriever (RRF) | Free |
| Text splitting | langchain-text-splitters | Free |
| Document loading | LangChain loaders (PDF, CSV, etc.) | Free |
| Evaluation | RAGAS | Free |
| Tracing (optional) | LangSmith Developer tier | Free (5k traces/mo) |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) local | Free |
| LLM (runtime) | OpenRouter API (qwen/qwen3.5-flash-02-23) | Pay per token |
| LLM (build-time) | Claude Code Haiku subagents | Free |
| Frontend | Streamlit or Gradio | Free |

No paid services needed: no LangGraph Platform, no LangSmith Plus, no Pinecone/Weaviate.

---

## Project structure

```
21-kmc-rag/
├── .env                          # OpenRouter API key (gitignored)
├── .gitignore
├── requirements.txt
├── project_plan.md               # this plan
├── architectural_decisions.md    # tradeoff log for interviews
├── unanswered_log.json           # questions the agent couldn't answer
│
├── data/
│   ├── inventory.csv             # master file list — single source of truth
│   ├── metadata/                 # per-document JSON metadata
│   │   └── {doc_id}.json
│   ├── processed_texts.json      # all cleaned text in one file {doc_id: {text, char_count}}
│   └── chunks.json               # serialized enriched chunks (intermediate artifact)
│
├── src/
│   ├── config.py                 # shared config (paths, model names, API settings)
│   ├── inventory.py              # Phase 1: walk folder, catalog, triage
│   ├── preprocess.py             # Phase 2: load, clean, extract text
│   ├── chunking.py               # Phase 3: split + contextual enrichment
│   ├── vectorstore.py            # Phase 4: embed, ChromaDB, hybrid search
│   └── agent.py                  # Phase 5: LangGraph router + tools
│
├── chroma_db/                    # ChromaDB persistence (gitignored)
│
├── evaluation/
│   ├── test_set.json             # evaluation questions with expected answers
│   └── ragas_results/            # RAGAS output per evaluation run
│
└── tests/                        # unit + integration tests
```

---

## Data flow — what the data looks like at each stage

This traces a single document (`Series A Strategy.pdf`) through the full pipeline.

### Stage 0: Raw file (source)

```
/Users/.../KeepMeCompany/5. Finances and VCs/Series A Strategy.pdf
```

Just the original file on disk. We never copy or move it.

### Stage 1: Inventory row (`data/inventory.csv`)

| doc_id | filename | file_type | file_size | char_count | path | status | include | content_type | sensitivity_flag | summary | topic_tags | audience | preprocessing_complete | extraction_quality | processed_char_count | chunking_complete | chunk_count |
|--------|----------|-----------|-----------|------------|------|--------|---------|--------------|-----------------|---------|------------|----------|----------------------|-------------------|---------------------|------------------|-------------|
| doc_017 | Series A Strategy.pdf | pdf | 245KB | 18420 | /Users/.../Series A Strategy.pdf | processed | yes | prose | no | Strategy document outlining... | fundraising,strategy,series-a | investors | yes | good | 17850 | yes | 12 |

One row per file. Columns accumulate across phases — starts sparse, fills up as we progress.

### Stage 1b: Metadata JSON (`data/metadata/doc_017.json`)

```json
{
  "doc_id": "doc_017",
  "filename": "Series A Strategy.pdf",
  "file_type": "pdf",
  "summary": "Strategy document outlining KMC's approach to Series A fundraising, including target investors, timeline, and key metrics to highlight.",
  "topic_tags": ["fundraising", "strategy", "series-a", "investors"],
  "audience": "investors",
  "date_created": "2024-03",
  "date_modified": "2024-05-12",
  "path_to_original": "/Users/.../Series A Strategy.pdf",
  "char_count": 18420
}
```

### Stage 2: Processed text (`data/processed_texts.json` — one entry)

```json
{
  "doc_017": {
    "text": "Series A Strategy\n\nOverview\nKMC is raising a Series A round of $3-5M to scale the platform from 12 pilot care homes to 200+ across the UK. Our core thesis: elderly care facilities need proactive health monitoring, not reactive incident response.\n\nTarget Investors\n- Health-tech focused VCs with portfolio companies in care/aging\n- Strategic investors from care home chains (HC-One, Four Seasons)\n...",
    "char_count": 17850
  }
}
```

Clean text. No page numbers, no PDF artifacts, no repeated headers. Single JSON file, one key per document.

### Stage 3: Enriched chunk (`data/chunks.json` — one entry)

```json
{
  "page_content": "From KMC's Series A fundraising strategy document (2024), in the section on target investors: Health-tech focused VCs with portfolio companies in care/aging. Strategic investors from care home chains (HC-One, Four Seasons). Angels with NHS or social care background.",
  "metadata": {
    "doc_id": "doc_017",
    "doc_name": "Series A Strategy.pdf",
    "summary": "Strategy document outlining KMC's approach to Series A...",
    "topic_tags": ["fundraising", "strategy", "series-a", "investors"],
    "audience": "investors",
    "date_created": "2024-03",
    "chunk_index": 3
  }
}
```

The `page_content` has the contextual prefix prepended. The `metadata` dict carries everything from Phase 1 plus chunk position.

### Stage 4: Vector in ChromaDB + BM25 index

```
ChromaDB stores:
  - vector: [0.0234, -0.1092, 0.0571, ...] (384 dimensions, MiniLM)
  - document: "From KMC's Series A fundraising strategy..."  (the page_content)
  - metadata: {doc_id: "doc_017", chunk_index: 3, ...}

BM25 index stores:
  - tokenized page_content for keyword matching
  - same metadata dict
```

Both retrievers return the same LangChain `Document` objects. EnsembleRetriever merges results via Reciprocal Rank Fusion.

### Stage 5: Query result (what the agent sees)

```
Query: "Who were KMC's target investors for Series A?"

EnsembleRetriever returns top-k Documents:
[
  Document(page_content="From KMC's Series A fundraising strategy...", metadata={...}),
  Document(page_content="From KMC's investor pitch deck (2024)...", metadata={...}),
  ...
]
```

The router agent receives these, evaluates quality, and either generates an answer or calls the doc specialist for more detail.

---

## Phase 1 — Corpus audit, inventory & metadata

**Goal**: Know exactly what's in the corpus. Every document gets cataloged, triaged, and enriched with metadata in a single pass.

### Source

Files at `/Users/eduardolizana/Documents/6. Companies /1. Companies with Rodrigo/KeepMeCompany/`. Referenced in place, not duplicated.

### Steps

1. **Walk the folder using LangChain's `DirectoryLoader`.** Recursively traverse the entire folder programmatically — no manual file-by-file listing. Output a CSV with columns:
   - `doc_id`: unique identifier
   - `filename`
   - `file_type` (pdf, xlsx, md, csv, txt, etc.)
   - `file_size`
   - `char_count`: character count of raw content (useful for chunking estimates later)
   - `path`: full path to original file
   - `status`: starts as "unprocessed"

2. **Automated triage via Haiku subagents.** Spawn subagents (3 in parallel) to read each file and propose:
   - `include`: yes/no/maybe (with reasoning)
   - `content_type`: prose / structured / mixed
   - `sensitivity_flag`: yes/no
   - Video files are auto-excluded without agent review.

3. **Generate metadata via Haiku subagents.** For each included file, generate:
   - `summary`: 2-3 sentence description
   - `topic_tags`: 3-5 tags (e.g., "fundraising", "technical-architecture", "compliance")
   - `audience`: investors / internal / board / personal
   - `date_created`: approximate
   - `date_modified`: from filesystem
   - `path_to_original`: full path for doc specialist
   - Subagents do a sniff test: flag anything that looks wrong or suspicious.

4. **Verification.** Count files in inventory vs actual files in folder (minus excluded video/media). They must match. Log any discrepancies.

### Output
A master `inventory.csv` with all columns above. Per-document metadata JSON files in `data/metadata/`. Updated `architectural_decisions.md` with any tradeoffs made.

---

## Phase 2 — Document preprocessing & format conversion

**Goal**: Get every included document into clean text ready for chunking.

### Loaders
- **PDFs** (all simple): `PyPDFLoader` from `langchain_community.document_loaders`
- **Markdown**: `TextLoader` — minimal processing needed
- **Spreadsheets**: Load with `CSVLoader` or `UnstructuredExcelLoader`. Add a contextual header from the metadata summary (generated in Phase 1). The doc specialist can load the full file on demand if deeper analysis is needed later.
- **Mixed folder**: `DirectoryLoader` with glob patterns to auto-route by file type

### Steps

1. **Load by file type.** Each loader returns LangChain `Document` objects with `page_content` and `metadata`.

2. **Clean extracted text.** Programmatic cleaning:
   - Remove repeated headers/footers
   - Remove page numbers
   - Collapse excessive whitespace
   - Remove PDF extraction artifacts
   - Preserve structural elements (headers, bullet points) that aid chunking

3. **Quality check.** Compare character count before and after cleaning. Flag documents where the delta is suspicious (> 20% loss). Log results in inventory.

4. **Store processed text** in `data/processed_texts.json` — single JSON file with `{doc_id: {text: "...", char_count: N}}`. Originals stay at source path.

5. **Update inventory**: `preprocessing_complete` (yes/no), `extraction_quality` (good/partial/needs-review), `processed_char_count`

### Output
`data/processed_texts.json` with all cleaned text. Inventory updated with preprocessing status.

---

## Phase 3 — Chunking & contextual enrichment

**Goal**: Turn processed documents into retrieval-ready chunks with contextual metadata.

### Tools
- `RecursiveCharacterTextSplitter` from `langchain_text_splitters` — default splitter, tries paragraph → sentence → word boundaries
- `MarkdownHeaderTextSplitter` — for markdown files, splits by headers preserving hierarchy

### Steps

1. **Chunk by document type (baseline config, tuned in Phase 4).**
   - **Prose documents**: Start with `RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)`. For markdown with clear headers, use `MarkdownHeaderTextSplitter` first, then recursive split on long sections.
   - **Spreadsheets**: chunk by logical sections (each table/data group = one chunk). Don't split tables across chunks.
   - These parameters are NOT final — they get tuned during Phase 4's iteration loop. Changing chunk_size means re-running this phase.

2. **Generate contextual enrichment via Haiku subagents (3 in parallel).** For each chunk, the subagent receives:
   - The file-level summary (from Phase 1 metadata)
   - The chunk itself
   - The text immediately before and after the chunk (surrounding context)

   It generates a short prefix that situates the chunk in its source document. Example:
   - Chunk: "The API latency averaged 340ms with a p95 of 620ms"
   - Prefix: "From KMC's technical performance report (Q3 2024), in the section on API benchmarks:"

   Prepend to `page_content` before embedding.

3. **Attach metadata to each chunk.** LangChain `Document.metadata` dict carries:
   - All fields from Phase 1 metadata (doc_id, date, audience, topic_tags, summary)
   - `chunk_index`: position in the document
   - `doc_name`: for the router to pass to the doc specialist if needed

4. **Update inventory**: `chunking_complete` (yes/no), `chunk_count` per document

### Output
Enriched chunks serialized to `data/chunks.json`. Inventory updated.

---

## Phase 4 — Embedding, vector store & retrieval tuning

**Goal**: Make chunks searchable and find the right retrieval parameters.

This phase has two parts: initial setup, then an iteration loop.

### Part A — Initial setup

1. **Set up ChromaDB.**
   ```python
   from langchain_chroma import Chroma
   from langchain_community.embeddings import SentenceTransformerEmbeddings

   embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
   vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
   vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
   ```

2. **Set up BM25.**
   ```python
   from langchain_community.retrievers import BM25Retriever

   bm25_retriever = BM25Retriever.from_documents(chunks, k=5)
   ```

3. **Combine into hybrid search.**
   ```python
   from langchain.retrievers import EnsembleRetriever

   ensemble_retriever = EnsembleRetriever(
       retrievers=[vector_retriever, bm25_retriever],
       weights=[0.6, 0.4]  # initial weights
   )
   ```

### Part B — Sniff test & test set

4. **Sniff test.** Pick 3-5 chunks with distinctive content. Query for that content and verify the right chunks come back. This catches obvious problems (bad embeddings, missing chunks, broken retrieval) before investing in a full test set.

5. **Build a test set (~20 questions).** Now that we can search, build questions informed by what's actually in the index:
   - Select 8-10 files from the inventory, spread across different topics and document types
   - For each file, identify facts uniquely contained there
   - Write 2-3 questions per file tied to specific chunks
   - Mix of: easy broad, precise, cross-document, and 3-4 "I don't know" questions
   - Store as `evaluation/test_set.json`

### Part C — Tuning loop (dedicated agent)

6. **Tuning.** Done iteratively — all decisions and experiments logged in `architectural_decisions.md`. Tuning covered: MMR vs similarity, cross-encoder reranking (reverted), three-stage pipeline (reverted), header stripping fix, section-aware prefixes. Current best: MMR-only hybrid at 87%.

### Output
A working, tuned hybrid retrieval system. Decisions and experiments logged in `architectural_decisions.md` for interview discussion.

---

## Phase 5 — Agent architecture (LangGraph)

**Goal**: Build the router agent and doc specialist sub-agent.

### High-level graph

```
[START] → Router → {RAG Search, Doc Specialist, I Don't Know} → Evaluate → {Retry → Router, Generate → [END]}
```

### Tools (all free, no sign-ups needed)
- `StateGraph`, `START`, `END` from `langgraph.graph`
- `ToolNode`, `tools_condition` from `langgraph.prebuilt`
- `InMemorySaver` from `langgraph.checkpoint.memory`
- `create_retriever_tool` from `langchain_core.tools`
- `@tool` decorator from `langchain_core.tools`
- LLM via OpenRouter API (OpenAI-compatible endpoint)

### Steps

1. **Define state schema.**
   ```python
   class AgentState(TypedDict):
       messages: Annotated[list, add_messages]
       documents: List[dict]
       quality_score: float
       retry_count: int
   ```

2. **Define tools.**
   - **RAG search**: wrap `ensemble_retriever` using `create_retriever_tool`
   - **Doc specialist**: `@tool` that takes (doc_name, question), loads full document, calls LLM via OpenRouter, returns answer
   - **I don't know**: `@tool` that logs question to `unanswered_log.json`, returns graceful message

3. **Build the graph.** Router → Tools → Evaluate → (Retry or Generate). Short prompts only — Edu refines later.

4. **Test each tool individually.**
   - Send a question to the RAG search tool directly, check results
   - Send a specific file + question to the doc specialist, check the answer
   - Send a question that should trigger "I don't know", verify it logs and responds correctly
   - Log findings in `architectural_decisions.md` — this informs how to structure data for the agent (e.g., if the agent struggles to find docs by name, we might need to change how doc_name is passed)

5. **Test the full pipeline** with the test set. End-to-end: question → routing → tool → evaluation → answer.

### Output
A working agent pipeline that takes a question and returns an answer. Architectural decisions logged from tool testing.

---

## Phase 6 — Evaluation (RAGAS)

**Goal**: Measure and document system quality.

### Steps

1. **Run RAGAS evaluation** on the test set:
   - Faithfulness: answers grounded in context?
   - Answer relevancy: answers address the question?
   - Context precision: retrieved chunks relevant?
   - Context recall: right chunks being retrieved?

2. **Identify failure modes.** Cross-document synthesis, precise numbers, temporal evolution, false confidence.

3. **Iterate.** Adjust chunking, retrieval weights, prompts based on results.

4. **Document results** in the README.

### Output
RAGAS scores documented. Known limitations identified.

---

## Phase 7 — Frontend & deployment

> **[AGENT: claude-frontend-0330 — DONE — handoff ready]**

**Goal**: Make it accessible via a link. This is straightforward if the pipeline works.

### Status

Steps 1 and 3 are complete. Steps 2 and 4 remain.

### What was built (2026-03-30)

**`app.py`** — two-tab Streamlit app, entry point is `streamlit run app.py`.

- **Chat tab**: question input with spinner, answer rendered with a collapsible Sources expander. Three starter prompt buttons shown before the first message, hidden after. Daily cap enforced before each agent call.
- **Evaluation tab**: loads `evaluation/retrieval_test_results.json` and renders accuracy metrics + full question table. Loads RAGAS results from `evaluation/ragas_results/` automatically if any `.json` or `.csv` files exist there (Phase 6 output drops straight in with no code changes needed).

**`src/call_cap.py`** — daily cap logic. Counter persists in `data/daily_calls.json` as `{"date": "...", "count": N}`, resets automatically on date change. Controlled via `MAX_DAILY_CALLS` env var (default 50). Tested: call 1 allowed, call 2 blocked, stale date resets correctly.

**`src/agent.py`** — added `ask_with_sources(question) -> {"answer": str, "sources": list[str]}` alongside the existing `ask()`. Parses `[Source: doc_name]` tags out of the LLM response text. `ask()` is untouched (RAGAS depends on it).

**Key decisions:**
- API key (`OPENROUTER_API_KEY`) is a server-side Streamlit Cloud secret — never exposed to the browser. Streamlit runs Python server-side.
- No streaming: `agent.invoke()` blocks until done; spinner used instead. Streaming is a v2 concern.
- No session persistence: stateless per tab, no DB needed for this use case.

### Remaining steps

1. **Deploy** on Streamlit Cloud (free tier, one-click from GitHub). Set `OPENROUTER_API_KEY` as a Streamlit Cloud secret. Optionally set `MAX_DAILY_CALLS`.

2. **Write the README.** Architecture, design decisions, evaluation results, how to run locally.

### Output
A deployed app with a shareable link. A comprehensive README.

---

## v2 ideas (not for now)

- Trace visualization panel (show retrieval + agent reasoning in real time)
- Query rewriting / HyDE for better retrieval on complex questions
- Reranking with a cross-encoder (sentence-transformers, free)
- Caching frequent queries
- Analytics on what hiring managers ask (from unanswered log)
- Upgrade embeddings to API-based (Voyage AI or OpenAI)
- Conversation memory across turns (LangGraph's InMemorySaver supports this)
