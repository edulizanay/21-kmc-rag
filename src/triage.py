# ABOUTME: Classifies corpus files via OpenRouter API (triage + metadata in one pass).
# ABOUTME: Processes files in parallel with incremental saves and retry logic.

import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).parent.parent / ".env")

client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    timeout=60.0,
)
MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3.5-flash-02-23")

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_PATH = DATA_DIR / "triage_results.json"

PROMPT = """You are classifying documents for a RAG corpus about KeepMeCompany, a health-tech startup building AI-powered phone agents for elderly care (NHS GP practices in the UK).

Classify this document and respond with ONLY a JSON object, no other text:

{{
  "include": "yes/no/maybe",
  "content_type": "prose/structured/mixed",
  "sensitivity_flag": "yes/no",
  "summary": "2-3 sentences describing the actual content",
  "topic_tags": ["tag1", "tag2", "tag3"],
  "audience": "investors/internal/board/personal/technical",
  "sniff_test_notes": "anything suspicious or notable, empty string if nothing"
}}

Rules:
- include=yes if useful for a RAG about the startup experience. no if empty, template, irrelevant, or pure raw data with no narrative.
- sensitivity_flag=yes if contains financial specifics, personal info, legal details
- summary must describe what the document ACTUALLY contains
- topic_tags: 3-5 tags

Document info:
- Filename: {filename}
- Type: {file_type}
- Folder: {folder}

Document content (first 5000 chars):
{text}
"""


async def classify_file(doc_id, row, text, semaphore, retries=2):
    async with semaphore:
        prompt = PROMPT.format(
            filename=row["filename"],
            file_type=row["file_type"],
            folder=row["folder"],
            text=text,
        )

        for attempt in range(retries + 1):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500,
                )
                raw = resp.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                result = json.loads(raw)
                result["_ok"] = True
                result["_tok"] = resp.usage.total_tokens if resp.usage else None
                return doc_id, result
            except json.JSONDecodeError:
                if attempt < retries:
                    await asyncio.sleep(2)
                    continue
                return doc_id, {"_raw": raw, "_ok": False}
            except Exception as e:
                if attempt < retries:
                    await asyncio.sleep(2)
                    continue
                return doc_id, {"_error": str(e), "_ok": False}


async def main():
    concurrency = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    with open(DATA_DIR / "extracted_texts.json") as f:
        texts = json.load(f)
    with open(DATA_DIR / "inventory.csv") as f:
        rows_list = list(csv.DictReader(f))
        rows = {r["doc_id"]: r for r in rows_list}

    # Filter bad files
    bad_ids = set()
    for r in rows_list:
        t = texts.get(r["doc_id"], "")
        if len(t) < 10 or t.startswith("PK\x03"):
            bad_ids.add(r["doc_id"])

    good_ids = [r["doc_id"] for r in rows_list if r["doc_id"] not in bad_ids]

    # Load existing results to skip already-done files
    existing = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            existing = json.load(f)

    todo = [d for d in good_ids if d not in existing]
    if limit > 0:
        todo = todo[:limit]

    print(
        f"Total good: {len(good_ids)} | Already done: {len(existing)} | To process: {len(todo)} | Concurrency: {concurrency}",
        flush=True,
    )

    if not todo:
        print("Nothing to do!", flush=True)
        return

    semaphore = asyncio.Semaphore(concurrency)
    all_results = dict(existing)
    ok = 0
    fail = 0
    start = time.time()

    tasks = [
        classify_file(d, rows[d], texts.get(d, "[No text]"), semaphore) for d in todo
    ]

    for i, coro in enumerate(asyncio.as_completed(tasks)):
        doc_id, result = await coro
        all_results[doc_id] = {k: v for k, v in result.items() if not k.startswith("_")}
        if result.get("_ok"):
            ok += 1
        else:
            fail += 1

        done = ok + fail
        if done % 10 == 0 or done == len(todo):
            elapsed = time.time() - start
            print(
                f"  {done}/{len(todo)} | OK:{ok} Fail:{fail} | {elapsed:.0f}s",
                flush=True,
            )

        # Incremental save every 25 files
        if done % 25 == 0 or done == len(todo):
            with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    print(
        f"\nDone! {ok}/{len(todo)} OK in {elapsed:.1f}s | Failures: {fail}", flush=True
    )

    includes = {}
    for r in all_results.values():
        inc = r.get("include", "unknown")
        includes[inc] = includes.get(inc, 0) + 1
    print(f"Include breakdown: {includes}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
