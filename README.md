# KMC RAG — Conversational Q&A over a Startup's Document Corpus

A RAG-powered chatbot that lets you explore KeepMeCompany's full history — product decisions, regulatory filings, sales strategy, clinical protocols, investor materials — by asking questions in plain English.

Built as a portfolio project to demonstrate end-to-end RAG system design, agentic orchestration with LangGraph, and production-style evaluation.

---

## How it works

```
YOUR QUESTION
      │
      ▼
┌─────────────┐
│   Router    │  Decides: is this answerable from chunks,
│   Agent     │  does it need a full document, or is it
│  (LangGraph)│  simply not in the corpus?
└──────┬──────┘
       │
   ┌───┴────────────────────┐
   │                        │
   ▼                        ▼
┌──────────┐        ┌───────────────┐
│   RAG    │        │ Doc Specialist│
│  Search  │        │  Sub-Agent    │
│          │        │               │
│ Vector   │        │ Loads the full│
│ + BM25   │        │ source file,  │
│ hybrid   │        │ answers from  │
│ retrieval│        │ complete text │
└────┬─────┘        └──────┬────────┘
     │                     │
     └──────────┬──────────┘
                │
                ▼
       ┌────────────────┐
       │    Evaluate    │  Good enough? → Generate answer
       │    Quality     │  Not good enough? → Retry
       └────────┬───────┘
                │
                ▼
          FINAL ANSWER
        (with source doc
           citations)
```

If the question has no answer in the corpus, the agent routes to an **"I don't know"** tool that logs the question to `unanswered_log.json` and returns a graceful response — no hallucination.

---

## The data pipeline

Everything starts as raw files (PDFs, Word docs, Excel spreadsheets, PowerPoints, Markdown, CSVs). Here's what happens to them:

```
RAW FILES (~500 docs, mixed formats)
         │
         │  Phase 1 — Inventory & triage
         │  Walk the folder, catalog every file,
         │  LLM classifies: include / exclude / maybe
         │  Generates summary, topic tags, audience
         ▼
INVENTORY.CSV + METADATA JSONs (one per doc)
         │
         │  Phase 2 — Text extraction & cleaning
         │  PDF → text, Excel → structured text,
         │  PPTX → slide text, DOCX → prose
         │  Remove headers/footers, page numbers,
         │  PDF artifacts
         ▼
PROCESSED_TEXTS.JSON (~351 docs, clean text)
         │
         │  Phase 3 — Chunking
         │  Markdown: split by headers (h1 > h2 > h3),
         │  header chain prepended back into chunk text
         │  Everything else: RecursiveCharacterTextSplitter
         │  chunk_size=512, overlap=50
         ▼
         │  Phase 3b — Contextual enrichment (Claude Haiku)
         │  Each chunk gets a one-sentence LLM-generated
         │  prefix describing what that specific chunk
         │  contains — not just the document title
         │
         │  e.g. instead of:
         │  "From EU Declaration of Conformity..."
         │
         │  it becomes:
         │  "From EU Declaration of Conformity...:
         │   Contact details for regulatory compliance
         │   officer Severin Hoegl and authorized
         │   representative Rodrigo Orpis."
         ▼
CHUNKS.JSON (11,710 enriched chunks)
         │
         │  Phase 4 — Embedding & indexing
         │  Embed with all-MiniLM-L6-v2 (384 dims, local)
         │  Store in ChromaDB with MMR retrieval
         │  Also build BM25 keyword index in parallel
         ▼
┌─────────────────────────────────────┐
│         HYBRID RETRIEVER            │
│                                     │
│  Vector search (0.6 weight)         │
│  catches semantic similarity        │
│  "what did seniors experience?"     │
│  → finds "elderly users reported"   │
│                                     │
│  BM25 keyword search (0.4 weight)   │
│  catches exact terms                │
│  "patient123" → finds asthma QOF code   │
│  "Severin Hoegl" → finds contact    │
│                                     │
│  Merged via Reciprocal Rank Fusion  │
└─────────────────────────────────────┘
```

---

## Corpus


| Stat                                       | Value                               |
| ------------------------------------------ | ----------------------------------- |
| Total files cataloged                      | 503                                 |
| Included in RAG                            | 351                                 |
| Excluded (private data, broken extraction) | 152                                 |
| Total chunks indexed                       | 11,710                              |
| File formats                               | PDF, DOCX, XLSX, PPTX, MD, CSV, TXT |
| Languages                                  | English, Spanish                    |


Document types span: regulatory filings (EU Declaration of Conformity, CE Marking), clinical protocols (Escalator disease database, Mediktor assessments), investor materials (pitch deck, VC emails, accelerator prep), sales collateral (ICB strategy, GP scripts), QOF business cases, engineering docs, and fine-tuning datasets.

---

## Key design decisions

### Why hybrid retrieval (vector + BM25)?

Vector search converts text into numerical representations (embeddings) and finds chunks with similar meaning — so a query about "elderly patients" can match a chunk that says "senior users". But it misses exact strings: a query for "patient123" or "99.5% uptime" won't match if those specific tokens aren't close in embedding space.

BM25 is a keyword search algorithm that scores documents by how often the query terms appear in them (weighted by rarity — common words like "the" count less than rare ones like "patient123"). It catches exact matches that vector search misses.

The two are combined using **Reciprocal Rank Fusion (RRF)**: each retriever returns its own ranked list, and RRF merges them by converting each rank into a score (1/rank) and summing across retrievers. This way a chunk that appears near the top of both lists gets boosted, while a chunk that only one retriever likes still has a chance. The vector retriever gets 0.6 weight and BM25 gets 0.4, reflecting that most questions are semantic but exact-match queries still need to work.

### Why MMR instead of plain similarity?

With 11,710 chunks, one specialist document can dominate all k retrieval slots for a given topic. The Escalator disease database alone has 17 chunks — without MMR, a query about the Escalator system returns 5 chunks from that one file, leaving no room for the summary document that mentions it in broader context. MMR (Maximal Marginal Relevance) actively penalises redundant chunks from the same source in favour of diverse results.

### Why contextual enrichment?

Without a prefix, a chunk like *"Live safety layer running during every call..."* has no retrieval signal for the query "what is the Escalator system?" — the word "Escalator" never appears in the text.

To fix this, each chunk was enriched with a one-sentence LLM-generated prefix using **Claude Haiku subagents**. The subagents read the full document context and generated a specific description of what each chunk contains — not just the document title, but the actual content. For example: *"From Summary.md, section: Products Built > 3. Escalator System (Real-Time Crisis Detection)"*.

This was especially important for markdown files where `MarkdownHeaderTextSplitter` strips headers into metadata by default — a bug that silently removes every section title from both the vector index and BM25.

### Why a Doc Specialist sub-agent?

Some questions can't be answered from chunks alone — particularly questions about spreadsheets. A query like "what are the revenue projections for March 2025?" requires understanding the full structure of a financial model: row headers, column headers, formulas, and the relationships between cells. Fetching a single chunk from an Excel file might return one cell's value while missing all the surrounding context needed to interpret it.

The Doc Specialist sub-agent handles this by loading the entire source document into memory and answering with full context. This also keeps the main agent's context window clean — it never reads file contents directly.

### Cross-encoder reranking — tried, reverted

The retrieval pipeline originally used plain hybrid search (vector + BM25). We noticed that certain specialist documents — the Escalator disease database, for example — dominated all retrieval slots for their topic, crowding out other relevant sources. Adding MMR fixed this by penalising redundant results and diversifying the output, pushing accuracy from ~65% to 87%.

We then tried adding a cross-encoder reranking step (fetch 50 candidates → cross-encoder rescores by relevance → MMR selects from the reranked list). The idea was to get better ranking within the diverse set. But in practice, the cross-encoder re-scored by pure relevance, pulling the dominant specialist document chunks right back to the top — undoing exactly the diversity that MMR had achieved. Accuracy dropped from 87% to 80%.

The takeaway: reranking works when your candidate pool has good diversity but poor internal ordering. In this corpus, document saturation is the primary problem, and MMR diversity matters more than fine-grained relevance scoring.

---

## Evaluation

### Retrieval benchmark

Tested with a 40-question set covering broad queries, precise single-document lookups, cross-document synthesis, and unanswerable questions.


| Run                        | Test set | Method                       | Score |
| -------------------------- | -------- | ---------------------------- | ----- |
| Initial baseline           | 20q      | Hybrid, k=5, no MMR          | ~65%  |
| After MMR + chunking fixes | 30q      | Hybrid + MMR, k=5            | 87%   |
| Cross-encoder experiment   | 30q      | Hybrid + cross-encoder + MMR | 80%   |
| Phase 6 (post enrichment)  | 40q      | Hybrid + MMR, k=5            | ~80%  |
| Phase 7 (retriever fixes)  | 40q      | Hybrid + MMR, k=8, k-capped  | ~85%  |


### RAGAS (end-to-end quality)

[RAGAS](https://docs.ragas.io/) evaluates the full RAG pipeline — not just whether the right chunks were retrieved, but whether the final answer is faithful, relevant, and well-grounded. It runs each test question through the agent, collects the answer and retrieved contexts, then uses a judge LLM to score four metrics.


**Phase 6 (baseline):**

| Metric            | Score | What it measures                                   |
| ----------------- | ----- | -------------------------------------------------- |
| Answer Relevancy  | 0.844 | Does the answer address the question?              |
| Faithfulness      | 0.587 | Is the answer grounded in retrieved context?       |
| Context Precision | 0.278 | Are the retrieved chunks relevant to the question? |
| Context Recall    | 0.300 | Were the right chunks retrieved (vs ground truth)? |

**Phase 7 (retriever fixes + prompt improvements):**

| Metric            | Score | Change    | What it measures                                   |
| ----------------- | ----- | --------- | -------------------------------------------------- |
| Answer Relevancy  | 0.782 | -0.062    | Does the answer address the question?              |
| Faithfulness      | 0.671 | +0.084    | Is the answer grounded in retrieved context?       |
| Context Precision | 0.275 | ~flat     | Are the retrieved chunks relevant to the question? |
| Context Recall    | 0.376 | +0.076    | Were the right chunks retrieved (vs ground truth)? |

**Reading the scores:** Faithfulness and recall both improved meaningfully. The answer relevancy dip is expected — the agent now correctly calls `i_dont_know` for unanswerable questions (IPO valuation, Series B targets, quantum computing) instead of fabricating answers; RAGAS scores these short refusals lower on relevancy. Context precision is roughly flat: the Phase 6 score was unreliable (most questions timed out during evaluation), while Phase 7 produces a real measurement for the first time — both are in the 0.275–0.278 range, suggesting precision was never as high as the Phase 6 partial score implied.

**What changed in Phase 7:**
- `k` increased from 5 → 8, vectorstore rebuilt with per-chunk enrichment applied
- `EnsembleRetriever` output capped at k (was returning up to 16 chunks instead of 8 — a pre-existing bug)
- Agent prompt strengthened: explicit `i_dont_know` trigger for clearly unanswerable questions, multi-search instruction for cross-document questions
- RAGAS evaluation timeout raised to 300s (context_precision requires ~34s per chunk × k sequential calls)

### Potential improvements

- **Tune chunk size and overlap** — the current 512/50 split is a baseline; smaller chunks may improve precision for fact-lookup queries.
- **Query expansion** — reformulating the user query into multiple variants before retrieval could improve recall.
- **Hybrid weight tuning** — the 0.6/0.4 vector/BM25 split was chosen as a reasonable default; systematic tuning on the test set could improve it.

---

## Tech stack


| Component                   | Tool                                    | What it does                                                 |
| --------------------------- | --------------------------------------- | ------------------------------------------------------------ |
| Orchestration               | LangGraph                               | Routes questions to the right tool, manages agent state      |
| Vector store                | ChromaDB (local)                        | Stores chunk embeddings, supports MMR retrieval              |
| Embeddings                  | all-MiniLM-L6-v2 (SentenceTransformers) | Converts text to 384-dim vectors, runs locally               |
| Keyword search              | BM25 (rank_bm25)                        | Scores chunks by query term frequency, catches exact matches |
| Hybrid merging              | EnsembleRetriever                       | Combines vector + BM25 results via Reciprocal Rank Fusion    |
| LLM (runtime)               | OpenRouter → qwen/qwen3.5-flash-02-23   | Generates answers, routes questions, judges quality          |
| LLM (build-time enrichment) | Claude Haiku subagents                  | Generated per-chunk contextual prefixes                      |
| Evaluation                  | RAGAS                                   | Measures faithfulness, relevancy, precision, recall          |
| Frontend                    | Streamlit                               | Chat interface + evaluation dashboard                        |


No paid infrastructure. Runs entirely locally except for the runtime LLM API calls.

---

## Running locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set your OpenRouter API key
echo "OPENROUTER_API_KEY=your_key_here" > .env

# Run the pipeline (chunking + embedding — skippable if chroma_db/ exists)
python src/chunking.py
python src/vectorstore.py

# Chat
streamlit run app.py
```

---

## Pending cleanup

> **Before deleting anything, verify the vectorstore rebuild completed successfully and the retrieval test scores look correct.**

The following files are safe to remove once the system is confirmed working:


| Path                                          | Why it exists                                               | Safe to delete when                               |
| --------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------- |
| `data/chunks.json.bak`                        | Backup taken before per-chunk enrichment merge              | Vectorstore rebuilt and test scores confirmed     |
| `data/extracted_texts.json`                   | Intermediate extraction artifact from early pipeline run    | Now — superseded by `processed_texts.json`        |
| `data/triage_results.json`                    | Raw LLM triage output before merging into inventory         | Now — superseded by `inventory.csv`               |
| `data/TRIAGE_BATCH_2_SUMMARY.md`              | One-off triage run summary                                  | Now — not part of project structure               |
| `data/TRIAGE_BATCH_2_VERIFICATION.txt`        | One-off triage verification artifact                        | Now — not part of project structure               |
| `data/_read_docs.py`                          | One-off script left in the wrong folder                     | Now — move to `src/` first if it's still needed   |
| `enrichment_staging/` (entire folder)         | Temporary staging files used during per-chunk enrichment    | Work is merged into `chunks.json`, safe to delete |
| `src/__pycache__/`, `evaluation/__pycache__/` | Python bytecode cache                                       | Any time — auto-regenerated on next run           |
| `evaluation/retrieval_test_results_bm25.json` | BM25-only test run (diagnostic, not the real hybrid result) | After full hybrid test results are saved          |


---

## Project structure

```
21-kmc-rag/
├── src/
│   ├── agent.py          # LangGraph router + tools
│   ├── vectorstore.py    # ChromaDB + hybrid retriever
│   ├── chunking.py       # Splitting + contextual enrichment
│   ├── preprocess.py     # Text extraction + cleaning
│   ├── inventory.py      # Corpus cataloging
│   └── config.py         # Shared paths and settings
├── data/
│   ├── inventory.csv     # Master file list
│   ├── metadata/         # Per-document JSON metadata
│   ├── chunks.json       # Enriched chunks (intermediate)
│   └── processed_texts.json
├── evaluation/
│   ├── test_set.json     # 40 evaluation questions
│   └── run_ragas.py      # RAGAS evaluation runner
├── chroma_db/            # Vector store (gitignored)
├── unanswered_log.json   # Questions the agent couldn't answer
└── architectural_decisions.md  # Full tradeoff log
```

