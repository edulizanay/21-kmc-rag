### Logs
- 2026-03-29: Used OpenRouter API (qwen3.5-flash) for triage+metadata instead of Claude Haiku subagents. Haiku subagents couldn't access Bash to extract text from binary files (docx/xlsx/pptx). OpenRouter gave structured JSON output, 15 parallel calls, 99% parse success rate, ~$1 total cost for 491 files.
- 2026-03-29: Pre-extracted text from all files before classification. Binary formats (docx, xlsx, pptx, pdf) need Python libraries; extracted once centrally via extract_text.py, then fed extracted text to LLM for classification.
- 2026-03-29: Excluded 12 files with private data (sales contact databases, CVs, patient consultation summaries). Excluded 12 files with broken extraction (empty docx/pdf, unsupported odt). Total corpus: 503 files, 351 included, 23 maybe, 129 excluded.
- 2026-03-29: odt format not supported by extract_text.py (reads as binary ZIP). Only 1 odt file in corpus, excluded. Would need odfpy library to support.
- 2026-03-30: Chunk size 512 tokens, overlap 50. Small chunks improve precision for specific facts; overlap prevents splitting mid-sentence. These are baseline params — tunable in Phase 4 if retrieval quality is low.
- 2026-03-30: Contextual enrichment — each chunk gets a 1-sentence LLM-generated prefix situating it in its source document (e.g., "From KMC's pitch deck, in the section on market size:"). Improves retrieval by adding document-level context to each chunk. Cost: ~11,690 OpenRouter calls (~$2).
- 2026-03-30: Hybrid retrieval: vector search (0.6 weight) + BM25 keyword search (0.4 weight) via EnsembleRetriever with Reciprocal Rank Fusion. Vector catches semantic similarity, BM25 catches exact terms (company names, QOF codes, medical acronyms). Weights are tunable.
- 2026-03-30: Embeddings: all-MiniLM-L6-v2 (384 dims, local, free). Good enough for a corpus this size. Avoids API cost/latency for embeddings. Upgrade path: swap to a larger model if retrieval quality is poor.
- 2026-03-30: Markdown files get header-aware splitting (MarkdownHeaderTextSplitter) before recursive size splitting. Preserves document structure. All other formats use RecursiveCharacterTextSplitter directly.
- 2026-03-30: Text cleaning flags docs with >20% character loss as "needs-review". Catches bad extractions without silently dropping content. Repeated lines (3+ occurrences, <100 chars) removed as likely headers/footers.
- 2026-03-30: Bug — MarkdownHeaderTextSplitter strips section headers from chunk content and puts them in metadata only. A chunk from "### 3. Escalator System (Real-Time Crisis Detection)" had its heading removed, leaving only "Live safety layer running during every call...". The word "Escalator" never appeared in the chunk text. Fixed by reconstructing the header chain (h1 > h2 > h3) and prepending it back into page_content before further splitting. Without this fix, every markdown section header is invisible to both the vector embedder and BM25 — queries using the section name return zero signal from that chunk.
- 2026-03-30: Bug — contextual prefix was generated from document-level summary only, producing the same generic prefix on all 53 chunks of Summary.md ("From Summary.md, internal resume raw material..."). This gave zero retrieval advantage to specific sections. Fixed by using the section header chain as the prefix when available ("From Summary.md, section: Products Built > 3. Escalator System (Real-Time Crisis Detection)."), falling back to summary only for non-markdown documents. The prefix now carries the section identity, not just the document identity.
- 2026-03-30: Retrieval failure diagnosis — with k=5, a specialist document (Escalator Architecture.docx, 9 chunks) saturated all retrieval slots for Escalator queries, preventing Summary.md from appearing in results at all even though it contained relevant content. Root cause: k=5 is too narrow when one document dominates a topic. Two fixes applied: (1) switched vector retriever to MMR (Maximum Marginal Relevance), which penalises redundancy — if doc_108 already has 2 chunks in the result set, MMR actively deprioritises a third chunk from doc_108 in favour of a chunk from a different document; (2) added fetch_k parameter so the vector retriever internally considers a larger candidate pool (fetch_k=20) before applying MMR to select the final k. BM25 is unaffected by MMR — it still returns its top-k by keyword score.
- 2026-03-30: On the bi-encoder vs cross-encoder distinction — current retrieval uses bi-encoders only (all-MiniLM-L6-v2). A bi-encoder converts query and each chunk into independent vectors at separate times; similarity is cosine distance between those vectors. Fast and scalable because chunks are pre-embedded at index time, but the model never sees query and chunk together. A cross-encoder (used in reranking) concatenates query + chunk as a single input and runs a full transformer pass over both simultaneously — attention can compare any word in the query against any word in the chunk directly. Far more accurate but cannot be pre-computed, so only viable on a small candidate set (20-50 docs), not the full 11,690-chunk corpus. Industry standard is bi-encoder retrieval for recall + cross-encoder reranking for precision. Not implemented yet — left as a Phase 4 tuning option if MMR + higher k is insufficient.
- 2026-03-30: Tried cross-encoder reranking on top of MMR — retrieval accuracy dropped from 80% to 55%. Root cause: MMR and cross-encoder reranking are solving opposite problems. MMR enforces diversity by actively deprioritising redundant chunks from the same document. The cross-encoder re-scores purely by relevance, which means it pulls the most relevant chunks back to the top — and those are almost always from the specialist document, undoing MMR's diversity. Effectively: MMR diversifies the candidate set, then the cross-encoder un-diversifies it. Reverted to MMR-only. The lesson is that reranking is the right tool when your candidate set has good diversity but poor internal ranking — it's the wrong tool when diversity itself is the problem you're trying to solve.
- 2026-03-30: Tried three-stage pipeline: wide retrieval (similarity, fetch_k=50) → cross-encoder reranking → manual MMR (lambda=0.7). Hypothesis: applying MMR *after* cross-encoder (instead of before) would preserve diversity while benefiting from cross-encoder precision. Result: 80% accuracy (24/30) vs 87% (26/30) for MMR-only on the same expanded test set. The three-stage pipeline lost 2 questions that MMR-only handled correctly — the cross-encoder's relevance scoring still biased toward specialist documents even when MMR was applied afterward, because the cross-encoder scores were concentrated in a narrow range for same-topic chunks, making MMR's diversity penalty insufficient to overcome them. Reverted to MMR-only. Conclusion: for this corpus where document saturation (not ranking quality) is the primary retrieval problem, MMR at retrieval time remains the best approach.
- 2026-03-30: Phase 5 LangGraph agent tested end-to-end. Three test cases: (1) broad question ("What did KMC do?") — returned a well-structured overview citing multiple documents; (2) precise question ("QOF codes for asthma?") — correctly identified AST007 with specific details from QOF - Business Case.xlsx; (3) unanswerable question ("IPO valuation?") — correctly identified no IPO info exists, but answered from context instead of calling the i_dont_know tool. The LLM prefers to synthesize a "no info found" answer from retrieved context rather than routing to the explicit "I don't know" tool. This is acceptable behaviour — the answer is correct — but means the unanswered_log.json won't capture borderline questions. Prompt tuning could fix this if needed.

---

## Retrieval Accuracy Benchmark Log

| Date       | Test Set | Method                        | Score        | Notes                                                                  |
|------------|----------|-------------------------------|--------------|------------------------------------------------------------------------|
| 2026-03-30 | 20q      | Hybrid (vector + BM25), k=5   | ~65% (13/20) | Initial baseline before MMR. Doc saturation dominated failures.        |
| 2026-03-30 | 30q      | Hybrid + MMR, k=5             | 87% (26/30)  | After MMR + header fix + section-aware prefix. Baseline confirmed.     |
| 2026-03-30 | 30q      | Hybrid + cross-encoder + MMR  | 80% (24/30)  | Three-stage pipeline (cross-encoder before MMR). Reverted.             |
| 2026-03-30 | 40q      | Hybrid + MMR, k=5             | 87% (35/40)  | Reported by system; test set had wrong label + duplicate. Unverified.  |
| 2026-03-30 | 40q (cleaned) | BM25-only, k=5           | 62% (25/40)  | Re-run after test set cleanup. BM25-only lower bound.                  |
| 2026-03-30 | 40q (cleaned) | BM25-only, k=8           | 70% (28/40)  | k=8 recovers 3 more: DEM004, doc_344 suppliers, pilot cross-doc.       |

---

## 2026-03-30: Failure Analysis — 87% on 40-Question Set (35/40 passed, 5 failed)

### Background
When the test set was expanded from 30 to 40 questions, the aggregate score held at 87% — meaning 5 questions failed. 4 of those 5 were likely the same failures carried forward from the 30-question set. The new 10 questions introduced ~1-2 new failures. BM25-only analysis (run directly against the chroma sqlite dump) and content inspection of the failing documents identified the following patterns.

**Important gap identified:** Pass/fail at the per-question level was not logged at each run — only the aggregate score was recorded. This made it hard to tell which specific questions were failing when the test set changed. Going forward, save the full `retrieval_test_results.json` file with a timestamp at every benchmark run, and reference it here.

### Failure Pattern 1 — Document Saturation at k=5 (Escalator topic, 2 questions)

Questions 26 and 36 both require doc_001 AND doc_107. doc_001 never surfaces because doc_108 (9 chunks) and doc_107 (17 chunks) fill all 5 retrieval slots for any Escalator-themed query. MMR helps somewhat — it would deprioritise a 3rd chunk from doc_108 in favour of doc_107 — but doc_001's single relevant chunk is still outscored. This is the same saturation bug documented earlier. It isn't fully fixed; MMR improved the 20→30 upgrade but doesn't solve it completely when the topic gap is large. Also: two near-identical questions in the test set means one failure counts as two, inflating the apparent miss rate.

*Diagnosis: k=5 structural limit + near-duplicate test questions.*
*Fix: increase k to 8; deduplicate the two Escalator cross-doc questions.*

### Failure Pattern 2 — Cross-Doc Query Asymmetry (2 questions)

The question "How did KeepMeCompany's EU regulatory classification compare to the NICE ESF tier system?" requires both doc_271 (EU Declaration) and doc_194 (NICE ESF). BM25 returns all 5 slots from doc_194 because "NICE Evidence Standards Framework", "tier", and "classification" are densely present in that document. doc_271 uses different vocabulary (MDR 2017/745, Class I, conformity) — none of which the query contains. The vector component helps semantically, but at k=5, doc_194 still dominates.

Similarly, the pitch deck + QOF cross-doc question ("What features did the pitch deck highlight and what QOF codes would they support?") splits vocabulary across a marketing document (doc_392) and a clinical coding spreadsheet (doc_021). Both partially score but k=5 can't guarantee both appear.

Root cause: cross-doc synthesis questions inherently have split vocabulary. A single hybrid query will always favour the document whose vocabulary aligns better with the query text. This is a fundamental limitation of single-query retrieval for multi-hop reasoning tasks. The right architecture is query decomposition: split the cross-doc question into sub-queries, retrieve independently for each, then merge results before generation.

*Diagnosis: k=5 + single-query retrieval for multi-hop questions.*
*Fix short-term: increase k to 8-10. Fix long-term: implement query decomposition in the LangGraph agent for questions that clearly span multiple domains.*

### Failure Pattern 3 — Wrong Test Label (1 question)

The question "What data retention and deletion policy did KeepMeCompany commit to in its CE marking documentation?" was written with doc_209 as the expected document. But doc_209 is the *Instructions for Use* patient-facing guide, which doesn't contain retention/deletion policy language. The document that actually covers data risk and policy is doc_212 (CE Marking Risk Analysis spreadsheet), which BM25 ranked first. The test expectation was wrong — this is a test quality issue, not a retrieval failure.

*Fix: update expected_doc_id for this question from doc_209 to doc_212, or rewrite the question to target doc_209's actual content (how to use the AI phone agent).*

### Summary

| Pattern                              | Questions Affected | Root Cause                                         |
|--------------------------------------|--------------------|----------------------------------------------------|
| Doc saturation (k=5)                 | 2 (questions 26, 36) | Escalator specialist docs fill all k slots       |
| Cross-doc query asymmetry            | 2                  | Single query can't retrieve both vocabulary profiles |
| Wrong test label                     | 1                  | doc_209 doesn't contain the expected content       |

### Next Actions
1. Increase k to 8 in evaluation runs — expected to resolve the saturation failures and partially fix cross-doc ones.
2. Fix the doc_209 test label. ✅ Done — repointed to doc_212 (CE Risk Analysis).
3. Deduplicate the two overlapping Escalator cross-doc questions. ✅ Done.
4. Log per-question results (not just aggregate %) at every future benchmark run.
5. Long-term: query decomposition in LangGraph for cross-doc synthesis questions.

---

## 2026-03-30: Prefix Quality Audit — 176/374 docs have saturated (all-identical) prefixes

### Finding
Running a prefix audit across all 11,710 chunks revealed:
- **197 docs (53%)** have good prefixes — meaningful LLM-generated summaries after the filename, giving each doc a distinct retrieval signal.
- **176 docs (47%)** are **fully saturated** — every single chunk in the document shares an identical prefix (first 120 chars). These are almost all large structured files (xlsx, csv, docx with many rows/entries): Agents Sandbox.xlsx (280 chunks, all identical), output.xlsx (270), output.csv (264), evaluated_conversations.xlsx (259), Analytics Tracking.xlsx (250), etc.
- Only **1 doc** had a truly generic short summary.

### Why this happens
For non-markdown documents, the prefix is generated from a document-level summary only (not per-section, since there are no markdown headers). A document with 280 chunks all get the same prefix: "From Agents Sandbox.xlsx (structured, technical): This Excel file contains engineering sandbox logs...". The prefix adds zero retrieval signal — every chunk looks identical to the retriever's prefix scanner. The chunk content itself still carries signal, but the prefix's job (situating the chunk in context) is completely wasted for these docs.

This is distinct from the original generic prefix bug (which was the same summary text repeated regardless of section). Here, the summary itself is good and specific — it's just that for a 280-row Excel file, every row chunk gets the same document-level description. The prefix correctly describes the document but does nothing to differentiate row 47 from row 203.

### Impact
The failing test questions all target docs that are in this saturated category:
- doc_271 (EU Declaration, 11 chunks, 1 unique prefix): correct answer is buried in chunk content but prefix gives no advantage
- doc_392 (Pitch Deck, 8 chunks, 1 unique prefix): same
- doc_344 (Supplier Checklist, 6 chunks, 1 unique prefix): same
- doc_383 (VC Emails, 12 chunks, 1 unique prefix): same
- doc_098 (Madrid demo, 4 chunks, 1 unique prefix): same

These are not large files (4-12 chunks) so per-row LLM prefix isn't the fix. For small docs, the real fix is **per-chunk LLM enrichment** — instead of generating one summary for the whole document and pasting it on every chunk, generate a 1-sentence description of what *that specific chunk* contains. E.g. for doc_271 chunk 3: "EU Declaration section listing the manufacturer's authorized representative and Severin Hoegl as regulatory contact." That's qualitatively different from stamping every chunk with "EU Declaration certifying Class I device under MDR 2017/745."

### Cost estimate for per-chunk enrichment
11,710 chunks × ~200 tokens per call (chunk + instruction) ≈ 2.3M tokens input + 11,710 × ~30 tokens output ≈ 350K tokens output. At Haiku pricing (~$0.25/M input, $1.25/M output): ~$0.58 input + $0.44 output ≈ **~$1 total**. Comparable to the original enrichment cost.

### Deeper audit results
After inspecting all 374 docs, the breakdown is:

| Category | Docs | Chunks | Issue |
|---|---|---|---|
| Non-saturated (varying prefixes per chunk) | 57 | ~600 | None — already have per-section context (e.g. doc_001 markdown) |
| Saturated + good doc summary (>100 chars) | 312 | ~11,000 | Doc-level summary is good, but every chunk gets the same prefix |
| Saturated + bad doc summary (<100 chars) | 5 | ~255 | Summary itself is truncated/incomplete AND all chunks identical |

**The enrichment was done correctly — it was LLM-generated from the start (chunk 1 through 11,710).** There is no code/LLM cutoff point. The problem is architectural: for non-markdown documents, the chunking pipeline had no section headers to use as per-chunk context, so it fell back to stamping the document-level summary on every chunk. This is the expected code-path behavior, not a partial run failure.

### Enrichment run — 2026-03-30
Spawned 14 parallel Haiku agents (13 small docs + doc_194 solo). Each agent read the staging file for its doc, generated a per-chunk one-sentence prefix using the full document summary as context, and wrote an enriched JSON. All 14 completed successfully: 337 chunks updated in chunks.json, 0 failures. Chunks.json backed up to chunks.json.bak before write.

Quality spot-check showed strong improvement: doc_271 chunk 1 now reads "Contact details for regulatory compliance officer Severin Hoegl and authorized representative Rodrigo Orpis..."; doc_392 chunk 0 now reads "Problem statement showing lonely seniors visit doctors for non-medical reasons, costing the UK healthcare system $4 billion annually"; doc_098 chunk 0 now reads "Madrid City Council virtual assistant Sofía initiates a social isolation check-in call with elderly client Sergio...".

BM25-only test after enrichment: k=5 went from 62%→58%, k=8 from 70%→60%. This is expected — BM25 is pure keyword overlap and the new descriptive prefixes add vocabulary that doesn't match query keywords. The real gain comes from the vector component, which will embed the semantically rich per-chunk descriptions much more specifically. Rebuilding the vectorstore is required to measure the true impact.

Two test label issues found during spot-check: (1) doc_051 test question asks for "QOF" objection-handling but the document contains email templates without the word QOF — the document may not actually address that question; (2) doc_212 question asks about "AI hallucination risk mitigation" but the doc's risks are RK1-RK5 (identity verification, cybersecurity, data corruption) — hallucination isn't mentioned. Both need re-evaluation.

### Recommendation
Two-tier fix, in order of impact:

**Tier 1 — Per-chunk re-enrichment for the 312 good-summary saturated docs (~$1):**
Keep the existing doc summary as context in the prompt, but ask the LLM to generate a 1-sentence description of *this specific chunk's content*. Example: instead of "From EU Declaration of Conformity... (certifying Class I device)" × 11 chunks, each chunk gets its own line like "EU Declaration section listing the manufacturer's authorized representative and Severin Hoegl as regulatory contact." The existing doc summary can be passed as `[document_summary]` in the prompt but the *output* should be chunk-specific.

**Tier 2 — Full re-enrichment for the 5 bad-summary docs (~negligible cost):**
doc_077, doc_205, doc_256, doc_407, doc_443 — summaries are truncated or cut off mid-sentence. These need both a new doc summary and per-chunk enrichment.

**What NOT to re-enrich:** The 57 non-saturated docs (mostly markdown, including doc_001) already have good per-chunk context via the section header chain. Re-enriching them would be wasteful and might actually reduce quality by replacing crisp section headers with LLM paraphrases.

---

## 2026-03-30: Phase 6 — RAGAS Evaluation Results

### Setup
- Judge LLM: same OpenRouter model (qwen3.5-flash) used by the agent
- Embeddings for AnswerRelevancy: local all-MiniLM-L6-v2 (same as retriever)
- 40 questions with hand-written ground truth answers sourced from actual documents
- Dataset cached after agent collection (40 agent calls); evaluation scores cached after RAGAS run (160 judge LLM calls)

### Results

| Metric | Score | What it measures |
|--------|-------|------------------|
| **Answer Relevancy** | 0.844 | Does the answer address the question? |
| **Faithfulness** | 0.587 | Is the answer grounded in retrieved context? |
| **Context Precision** | 0.278 | Are the retrieved chunks relevant to the question? |
| **Context Recall** | 0.300 | Were the right chunks retrieved (vs ground truth)? |

### Interpretation
- **Answer relevancy (0.844)** is the strongest metric — the LangGraph agent generates relevant, well-structured answers when it has context.
- **Faithfulness (0.587)** is moderate — the agent sometimes synthesizes beyond what the retrieved chunks contain, drawing on LLM knowledge rather than strictly grounding in context. This is expected behavior for a conversational agent but would be a concern for a strictly-grounded QA system.
- **Context precision (0.278) and recall (0.300)** confirm that retrieval is the bottleneck, not generation. The retriever frequently returns chunks from the wrong documents or misses the target documents entirely. This is consistent with the retrieval benchmark (87% at k=5, dropping for cross-doc and precise queries).
- ~15 OpenRouter TimeoutErrors during the 160-call evaluation may have produced NaN scores for some samples, slightly dragging down averages.

### Key takeaway
Improving retrieval quality (better chunk prefixes, higher k, query decomposition for cross-doc questions) will have the highest ROI on overall RAG quality. The agent's generation layer is already performing well.