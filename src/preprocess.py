# ABOUTME: Phase 2 — loads included documents, cleans extracted text, and stores processed output.
# ABOUTME: Produces data/processed_texts.json with clean text ready for chunking.

import csv
import json
import re

from src.config import DATA_DIR, INVENTORY_PATH, PROCESSED_TEXTS_PATH


def load_included_docs() -> list[dict]:
    """Load inventory rows for included (yes/maybe) documents."""
    with open(INVENTORY_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["include"] in ("yes", "maybe")]


def clean_text(text: str) -> str:
    """Clean extracted text for chunking readiness."""
    # Remove null bytes and control characters (except newlines/tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    # Collapse runs of 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse runs of 3+ spaces into 1
    text = re.sub(r" {3,}", " ", text)

    # Remove repeated headers/footers (same line appearing 3+ times)
    lines = text.split("\n")
    line_counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if stripped:
            line_counts[stripped] = line_counts.get(stripped, 0) + 1

    repeated = {line for line, count in line_counts.items() if count >= 3 and len(line) < 100}
    if repeated:
        lines = [line for line in lines if line.strip() not in repeated]
        text = "\n".join(lines)

    # Remove page number patterns (standalone numbers or "Page X of Y")
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"\n\s*Page \d+ of \d+\s*\n", "\n", text, flags=re.IGNORECASE)

    # Remove common PDF artifacts
    text = re.sub(r"\f", "\n", text)  # form feeds

    # Final whitespace cleanup
    text = text.strip()

    return text


def assess_quality(raw_len: int, clean_len: int) -> str:
    """Assess extraction quality based on character count delta."""
    if clean_len == 0:
        return "empty"
    if raw_len == 0:
        return "needs-review"
    delta = (raw_len - clean_len) / raw_len
    if delta > 0.20:
        return "needs-review"
    return "good"


def preprocess_all() -> dict:
    """Load, clean, and store all included documents."""
    included = load_included_docs()
    print(f"Included documents: {len(included)}")

    # Load raw extracted texts
    extracted_path = DATA_DIR / "extracted_texts.json"
    with open(extracted_path, encoding="utf-8") as f:
        raw_texts = json.load(f)

    # Also load full text for included docs (extracted_texts has 5000 char limit)
    # Re-extract without limit for included docs
    from src.extract_text import extract_text

    processed = {}
    quality_counts = {"good": 0, "needs-review": 0, "empty": 0}
    inventory_updates = {}

    for row in included:
        doc_id = row["doc_id"]
        filepath = row["path"]

        # Extract full text (no truncation)
        raw = extract_text(filepath, max_chars=100_000)
        raw_len = len(raw)

        cleaned = clean_text(raw)
        clean_len = len(cleaned)

        quality = assess_quality(raw_len, clean_len)
        quality_counts[quality] = quality_counts.get(quality, 0) + 1

        if clean_len > 0:
            processed[doc_id] = {
                "text": cleaned,
                "char_count": clean_len,
            }

        inventory_updates[doc_id] = {
            "preprocessing_complete": "yes",
            "extraction_quality": quality,
            "processed_char_count": str(clean_len),
        }

    # Write processed texts
    with open(PROCESSED_TEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False)
    print(f"Wrote {len(processed)} processed texts to {PROCESSED_TEXTS_PATH}")
    print(f"Quality: {quality_counts}")

    # Update inventory
    _update_inventory(inventory_updates)

    return processed


def _update_inventory(updates: dict) -> None:
    """Update inventory.csv with preprocessing results."""
    with open(INVENTORY_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    for row in rows:
        if row["doc_id"] in updates:
            row.update(updates[row["doc_id"]])

    with open(INVENTORY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {len(updates)} rows in inventory.csv")


if __name__ == "__main__":
    preprocess_all()
