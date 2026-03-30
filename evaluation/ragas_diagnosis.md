# RAGAS Evaluation Diagnosis
## KeepMeCompany RAG System — Retrieval Bottleneck Analysis

**Date:** 2026-03-30
**Dataset:** 40 questions with ground truth
**Purpose:** Identify highest-ROI fixes for context_precision (0.278) and context_recall (0.300)

---

## Overall Scores

| Metric | Score | Status |
|--------|-------|--------|
| Answer Relevancy | 0.844 | Strong — generation layer works well |
| Faithfulness | 0.587 | Moderate — some hallucination when context is weak |
| **Context Precision** | **0.278** | **Primary bottleneck** |
| **Context Recall** | **0.300** | **Primary bottleneck** |

**Key finding:** The generation layer is healthy (relevancy 0.84). The retrieval layer is the bottleneck. Fixing retrieval will have cascading benefits on faithfulness as well.

---

## Task 1: Worst-Performing Questions

### Bottom 10 by Context Recall

| Rank | Question | Recall | Precision | Faithfulness | Expected Doc(s) |
|------|----------|--------|-----------|--------------|-----------------|
| 1 | Who was responsible for regulatory compliance at KeepMeCompany? | 0.0 | — | 0.875 | doc_271 |
| 2 | What was KeepMeCompany's subscription price and expected margin? | 0.0 | — | 0.312 | doc_392 |
| 3 | What tech stack did KeepMeCompany use for their backend? | 0.0 | 0.0 | 0.350 | doc_001 |
| 4 | What is the NICE Evidence Standards Framework classification system? | 0.0 | — | 0.500 | doc_194 |
| 5 | What were the founders' backgrounds before KeepMeCompany? | 0.0 | — | 0.000 | doc_392, doc_001 |
| 6 | What improvements did the AI intake system achieve for patient wait times? | 0.0 | — | 0.000 | doc_001 |
| 7 | What diseases did the Mediktor assessment identify for a patient with runny nose? | 0.0 | — | 0.000 | doc_473 |
| 8 | What was KeepMeCompany's adoption strategy for getting customers? | 0.0 | — | 0.607 | doc_392 |
| 9 | How does quantum computing relate to KeepMeCompany's technology? | 0.0 | — | 0.000 | *(none — unanswerable)* |
| 10 | What AI infrastructure providers did KeepMeCompany use per Supplier Checklist? | 0.0 | 0.0 | 0.000 | doc_344 |

### Bottom 5 by Faithfulness (distinct from above)

| Rank | Question | Faithfulness | Recall | Expected Doc(s) |
|------|----------|--------------|--------|-----------------|
| 1 | How did KeepMeCompany's AI companion interact with lonely seniors in Madrid? | 0.0 | 0.667 | doc_098 |
| 2 | What was KeepMeCompany's brochure pitch for private clinics in Spain? | 0.0 | 0.667 | doc_035 |
| 3 | What specific NHS access metric example is cited in the ICB sales strategy doc? | 0.0 | 0.0 | doc_049 |
| 4 | What was the name of the AI assistant used in the Madrid city council demo? | 0.0 | 0.0 | doc_098 |
| 5 | What were the pilot results + QOF business case projected income? | 0.0 | 0.0 | doc_001, doc_021 |

---

## Task 2: Retrieval Cross-Reference (Bottom 10 Combined Score)

For each question, expected doc IDs vs. actually retrieved doc IDs:

| # | Question (abbreviated) | Expected Docs | Retrieved Docs | Doc Hit? | Category |
|---|------------------------|---------------|----------------|----------|----------|
| 1 | Regulatory compliance responsibility | doc_271 | doc_205, doc_244, doc_223, doc_343, doc_292 | **NO** | A: Retrieval miss |
| 2 | Subscription price and margin | doc_392 | doc_255, **doc_392**, doc_367, doc_372, doc_176 | YES | C: Faithfulness — wrong chunk |
| 3 | Tech stack (backend) | doc_001 | doc_115, doc_394, doc_383, doc_378 | **NO** | A: Retrieval miss |
| 4 | NICE ESF classification system | doc_194 | doc_192, **doc_194** | YES | C: Wrong chunk within doc |
| 5 | Founders' backgrounds | doc_392, doc_001 | doc_378, doc_115, doc_383, doc_131, doc_447 | **NO** | A: Retrieval miss (both docs) |
| 6 | AI intake system — wait time improvements | doc_001 | doc_447, doc_041, doc_413, doc_354, doc_404 | **NO** | A: Retrieval miss |
| 7 | Mediktor assessment — runny nose | doc_473 | **doc_473**, doc_329, doc_472, doc_475, doc_471 | YES | C: Faithfulness — agent hallucinated |
| 8 | Adoption strategy | doc_392 | doc_445, doc_383, doc_121, doc_257, doc_376 | **NO** | A: Retrieval miss |
| 9 | Quantum computing (unanswerable) | *(none)* | doc_372, doc_115, doc_018, doc_479, doc_250 | N/A | D: Unanswerable — should refuse |
| 10 | AI infrastructure providers (Supplier Checklist) | doc_344 | doc_293, doc_244, doc_415, doc_394, doc_342 | **NO** | A: Retrieval miss |

### Key observation
Questions 2 and 4 retrieved the right document but RAGAS scored recall=0.0. This suggests RAGAS is checking for specific chunks (not just doc presence) and the retrieved chunks from the correct doc didn't contain the answer passage. This is a **chunk boundary / granularity problem**, not a document-level miss.

---

## Task 3: Failure Pattern Categorization

### Distribution

| Pattern | Count | % | Description |
|---------|-------|---|-------------|
| **A: Retrieval miss** | 9 | 22.5% | Expected doc not retrieved at all |
| **B: Chunk boundary** | ~2 | 5% | Right doc retrieved but wrong chunk (questions 2, 4) |
| **C: Faithfulness** | 11 | 27.5% | Context retrieved but answer not grounded in it |
| **D: Unanswerable** | 8 | 20% | Question has no answer in corpus; agent should refuse |
| **E: Passing/marginal** | 10 | 25% | Good or near-passing performance |

### Pattern A — Retrieval Miss (9 questions)

The retrieval system completely fails to surface the expected document:

- **doc_001 (Summary.md)** missed for 4+ questions: "tech stack", "founders' backgrounds", "AI intake wait times", "pilot results + QOF income"
- **doc_392 (Pitch Deck)** missed for 3 questions: "subscription price", "founders' backgrounds", "adoption strategy"
- **doc_344 (Supplier Checklist)** missed for 1 question: "AI infrastructure providers"
- **doc_271 (EU Declaration of Conformity)** missed for 1 question: "regulatory compliance responsibility"

Root cause for doc_001: Summary.md has 73 chunks — many with similar generic prefixes. MMR deprioritizes these as "redundant". The question uses vocabulary like "tech stack" or "backend" but the chunk content may use different terminology.

Root cause for doc_392: Pitch Deck has only 8 chunks but uses investor/marketing vocabulary ("adoption strategy", "subscription model") that doesn't match how the question is phrased.

Root cause for doc_344/doc_271: Small specialist documents (6 and 11 chunks) with narrow vocabulary. Question uses specific terminology ("Supplier Qualification Checklist", "EU Declaration") that may not appear verbatim in the top-k candidates.

### Pattern C — Faithfulness Issues (11 questions)

Right documents were retrieved but the agent's answer wasn't grounded in the retrieved passages. Two sub-types:

- **C1: Agent ignores context and uses LLM knowledge** — Madrid demo questions (doc_098, doc_035): the right Spanish brochure content was retrieved (recall=0.67) but the agent answered with generic AI companion descriptions rather than the specific text. The context contained the answer but the agent didn't use it.
- **C2: Chunk retrieved but answer not in that chunk** — Questions 2, 4: doc_392 and doc_194 were in the retrieved set but the specific chunk containing the answer wasn't (RAGAS recall = 0.0 despite doc presence).

### Pattern D — Unanswerable Questions (8 questions)

Questions the corpus cannot answer. Examples:
- "How does quantum computing relate to KeepMeCompany's technology?" — no such content exists
- "What was KeepMeCompany's IPO valuation?" — company never IPO'd
- "What was KeepMeCompany's Series B fundraising target?" — no such document

RAGAS scores these as near-zero because the agent retrieved random documents and attempted an answer instead of refusing. This is an **agent behavior problem** (should say "I don't have information on this"), not a retrieval problem. These 8 questions are unlikely to benefit from retrieval improvements.

---

## Task 4: Chunk Quality Analysis for Failing Documents

### doc_001 (Summary.md) — 73 chunks, missed in ~4 questions

**Sample chunk:**
```
From Summary.md, section: KeepMeCompany — Resume Raw Material.

KeepMeCompany — Resume Raw Material

Everything extracted from the company's docs and codebase. Use this as a source of truth...
```

**Problems:**
- Many chunks share near-identical section-level prefixes ("From Summary.md, section: X")
- MMR treats these as redundant and deprioritizes them after retrieving the first one
- Chunks covering tech stack, pilot results, and AI intake are buried in a large document
- The document is a "kitchen sink" dump — specific facts are hard to retrieve precisely

**Fixable?** Per-chunk enrichment (with specific-fact prefixes rather than generic section headers) would help. Also, increasing k from 5→8 would recover some of these since the doc has high coverage.

### doc_392 (Pitch Deck) — 8 chunks, missed in ~3 questions

**Sample chunk:**
```
From KeepMeCompany - Pitch Deck.pptx (structured, investors): Problem statement showing that
lonely seniors visit doctors for non-medical reasons, costing the UK healthcare system $4B annually.
```

**Problems:**
- Uses investor/marketing vocabulary: "adoption strategy", "go-to-market", "unit economics"
- Questions use similar vocabulary but there's a vocabulary split — "subscription price and expected margin" vs. chunk text that uses "£X per patient per month" without using the word "margin"
- Only 8 chunks — low probability of hitting the right one with k=5 and hybrid retrieval competing for slots

**Fixable?** Query expansion or per-chunk enrichment for financial/go-to-market slides. Also a good candidate for k increase.

### doc_271 (EU Declaration of Conformity) — 11 chunks, missed in 1 question

**Sample chunk:**
```
From EU Declaration of Conformity - 1_13_25, 12_56 PM.pdf (mixed, internal): Formal declaration
statement identifying KeepMeCompany Ltd at London address as manufacturer...
```

**Problem:** Question asks about "regulatory compliance responsibility" (a person's name), but the chunk text describes the document's legal structure, not the individual named in it. The answer is likely in a different chunk that lists the named responsible person. The retrieval hit the right document but not the right chunk — Pattern B.

**Fixable?** More granular chunking of this document or a targeted enrichment prefix mentioning the named person.

### doc_344 (Supplier Qualification Checklist) — 6 chunks, missed in 1 question

**Sample chunk:**
```
From Supplier Qualification Checklist.docx (mixed, internal): Defines the scope and purpose
of the supplier qualification checklist used for initial evaluations and re-evaluations...
```

**Problem:** Question asks "What AI infrastructure providers did KeepMeCompany use *according to the Supplier Qualification Checklist*?" The document title appears in the question but the retrieved chunks were from unrelated docs. The current per-document prefix ("Supplier Qualification Checklist.docx") should match — but with k=5 and many competing chunks, it lost out.

**Fixable?** This is a k problem. Increasing to k=8 likely recovers this since the question literally names the document.

---

## Task 5: Architectural History — What Was Already Tried

*From architectural_decisions.md — do not re-propose these.*

| Approach | Outcome | Notes |
|----------|---------|-------|
| MMR at retrieval time (fetch_k=20) | **KEPT** | Core improvement; 65%→87% baseline |
| Header-aware chunking + header reconstruction | **KEPT** | Critical fix — headers now visible to embedder and BM25 |
| Contextual enrichment (document-level prefixes) | **KEPT (partial)** | Works for markdown; creates saturation for non-markdown |
| Hybrid retrieval (vector + BM25, 0.6/0.4) | **KEPT** | Semantic + exact-term coverage |
| Cross-encoder reranking (full pipeline) | **REVERTED** | 80%→55%; undid MMR diversity gains |
| Three-stage pipeline (wide→cross-encoder→MMR) | **REVERTED** | 80% vs 87% MMR-only |
| Per-chunk enrichment (337 of 11,710 chunks updated) | **PARTIAL** | Vectorstore not yet rebuilt with new embeddings |

**What has NOT been tried:**
- Query decomposition for cross-document synthesis questions
- Increasing k from 5 to 8 (tested in BM25 only, not in production hybrid retrieval)
- Lightweight cross-encoder on final top-k *after* MMR (different from the reverted approach)
- Synthetic query expansion / reformulation
- Document routing by domain type

---

## Priority Action Plan

### Priority 1: Rebuild Vectorstore with Per-Chunk Enrichment
**Impact estimate:** +3–5 questions (7–12%)
**Effort:** Medium (1–2 days)
**Affected questions:** All 9 Pattern A misses, especially doc_001 and doc_392 questions

Per-chunk enrichment was applied to 337 chunks (14 specialist documents including doc_194), but the vectorstore was never rebuilt. The old embeddings don't reflect the new prefixes. Rebuilding will apply the specificity gains that per-chunk enrichment promises for semantic retrieval.

**Validation:** Re-run RAGAS on questions 3, 5, 6, 11 (doc_001 failures) + questions 2, 8 (doc_392 failures) = ~6 questions instead of 40. That's 85%+ cost savings.

**Questions to test:** Q3 (tech stack), Q5 (founders), Q6 (AI intake wait times), Q8 (adoption strategy), Q2 (subscription price), Q11 (NICE ESF)

---

### Priority 2: Increase k from 5 to 8
**Impact estimate:** +1–3 questions (2–7%)
**Effort:** Low (1-line config change)
**Affected questions:** doc_344 (Supplier Checklist), doc_271 (EU Declaration), any question where the right doc was in position 6–8

BM25-only testing showed k=8 recovers 3 additional questions over k=5 (62%→70%). The same pattern likely applies to hybrid retrieval. Small documents (6–11 chunks) are underrepresented at k=5 when competing against high-volume docs.

**Validation:** Re-run on Q10 (AI infrastructure providers), Q1 (regulatory compliance), Q4 (NICE ESF classification) = 3 questions.

**Risk:** Slightly higher LLM cost (8 chunks vs 5 in context). Faithfulness may dip slightly if more irrelevant chunks compete.

---

### Priority 3: Query Decomposition for Cross-Doc Synthesis Questions
**Impact estimate:** +2–3 questions (5–7%)
**Effort:** Medium (requires agent-level change)
**Affected questions:** Q5 (founders — needs doc_392 AND doc_001), Q25 (pilot results + QOF income — needs doc_001 AND doc_021)

Multi-hop questions that require combining facts from different documents fail because a single query can't reliably surface both documents. The vocabulary in one document doesn't match the vocabulary in the other. Solution: detect cross-doc questions at the agent level, decompose into sub-queries, retrieve separately, merge before generation.

**Validation:** Re-run on Q5 (founders' backgrounds), Q25 (pilot + QOF), and similar cross-doc questions = 2–3 questions.

**Note:** This requires careful design to avoid over-decomposing simple questions. Apply only when expected_doc_ids in test_set shows 2+ distinct document roots.

---

### Priority 4: Audit and Fix Test Set Labels
**Impact estimate:** +1–2 questions (automatic wins)
**Effort:** Low (manual verification of ~5 questions)
**Affected questions:** Any where expected_doc_ids points to the wrong document

At least one label error is suspected: a question where the expected doc doesn't actually contain the answer. These score as failures even when retrieval is correct. Before evaluating improvements, verify that the bottom 10 expected_doc_ids are actually correct by checking those specific chunks in chunks.json.

**Validation:** Zero cost — these become automatic passes once labels are fixed. Cross-check Q1, Q2, Q4 expected docs against ground truth answers in ground_truth.json.

---

### Priority 5: Improve "I Don't Know" Behavior for Unanswerable Questions
**Impact estimate:** +2–4 faithfulness points (not precision/recall)
**Effort:** Low (agent prompt change)
**Affected questions:** 8 unanswerable questions (Q9: quantum computing, IPO valuation, Series B, etc.)

These questions have no answer in the corpus. The agent currently retrieves random documents and fabricates an answer, scoring faithfulness=0. Adding explicit "I don't have information on this" behavior when retrieved context is low-confidence would eliminate these false failures and improve faithfulness score.

**Validation:** Re-run on the 8 unanswerable questions only. Expected: faithfulness improves from 0 to ~1.0 for these.

**Note:** This doesn't improve context_precision or context_recall (RAGAS treats empty expected_docs specially) but fixes the faithfulness floor.

---

## Cheap Validation Strategy

The RAGAS evaluation makes ~160 LLM calls for 40 questions (4 metrics × 40). Re-running all 40 costs ~$X and takes time. Instead:

| Action | Subset | Questions | Cost vs full run |
|--------|--------|-----------|-----------------|
| Priority 1 (rebuild vectorstore) | Q2, Q3, Q5, Q6, Q8, Q11 | 6 questions | ~15% |
| Priority 2 (k=8) | Q1, Q4, Q10 | 3 questions | ~8% |
| Priority 3 (query decomp) | Q5, Q25 | 2 questions | ~5% |
| Priority 4 (label audit) | Q1, Q2, Q4 | 3 questions | ~8% |
| Priority 5 (unanswerable) | 8 unanswerable questions | 8 questions | ~20% |
| Full re-eval after all changes | All 40 | 40 questions | 100% |

**Recommended sequence:**
1. Fix labels (Priority 4) — zero cost, do first
2. Increase k (Priority 2) — 1-line change, test on 3 questions
3. Rebuild vectorstore (Priority 1) — test on 6 questions
4. Query decomposition (Priority 3) — test on 2 questions
5. Full 40-question re-eval after all changes pass subset tests

---

## Most Frequently Missed Documents Summary

| Doc ID | Filename | Chunks | Questions Affected | Primary Fix |
|--------|----------|--------|-------------------|-------------|
| doc_001 | Summary.md | 73 | 4+ questions | Per-chunk enrichment + rebuild vectorstore |
| doc_392 | KeepMeCompany - Pitch Deck.pptx | 8 | 3 questions | Per-chunk enrichment for financial slides + k=8 |
| doc_271 | EU Declaration of Conformity.pdf | 11 | 1 question | k=8 (small doc competing against large ones) |
| doc_194 | esf-classification-examples.xlsx | 153 | 1 question | Rebuild vectorstore (per-chunk enrichment already applied) |
| doc_344 | Supplier Qualification Checklist.docx | 6 | 1 question | k=8 (very small doc, literal doc name in question) |
