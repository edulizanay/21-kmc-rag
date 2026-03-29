# ABOUTME: Merges triage results into inventory.csv and writes per-doc metadata JSONs.
# ABOUTME: Run after triage.py finishes classifying all files via OpenRouter.

import csv
import json

from src.config import DATA_DIR, INVENTORY_PATH, METADATA_DIR

TRIAGE_RESULTS_PATH = DATA_DIR / "triage_results.json"


def load_triage_results() -> dict:
    """Load triage results from the consolidated JSON file."""
    with open(TRIAGE_RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} triage entries from {TRIAGE_RESULTS_PATH.name}")
    return data


def update_inventory(triage_data: dict) -> None:
    """Update inventory.csv with triage+metadata fields."""
    with open(INVENTORY_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    updated = 0
    for row in rows:
        doc_id = row["doc_id"]
        if doc_id in triage_data:
            entry = triage_data[doc_id]
            tags = entry.get("topic_tags", [])
            if isinstance(tags, list):
                tags = ",".join(tags)
            row["include"] = entry.get("include", "")
            row["content_type"] = entry.get("content_type", "")
            row["sensitivity_flag"] = entry.get("sensitivity_flag", "")
            row["summary"] = entry.get("summary", "")
            row["topic_tags"] = tags
            row["audience"] = entry.get("audience", "")
            row["status"] = "triaged"
            updated += 1

    with open(INVENTORY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {updated}/{len(rows)} rows in inventory.csv")


def write_metadata_jsons(triage_data: dict, rows: list[dict]) -> None:
    """Write per-document metadata JSON files."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    rows_by_id = {r["doc_id"]: r for r in rows}
    written = 0

    for doc_id, entry in triage_data.items():
        if entry.get("include") == "no":
            continue

        row = rows_by_id.get(doc_id, {})
        tags = entry.get("topic_tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]

        metadata = {
            "doc_id": doc_id,
            "filename": row.get("filename", ""),
            "file_type": row.get("file_type", ""),
            "summary": entry.get("summary", ""),
            "topic_tags": tags,
            "audience": entry.get("audience", ""),
            "path_to_original": row.get("path", ""),
            "file_size": row.get("file_size", ""),
            "folder": row.get("folder", ""),
            "content_type": entry.get("content_type", ""),
            "sensitivity_flag": entry.get("sensitivity_flag", ""),
            "sniff_test_notes": entry.get("sniff_test_notes", ""),
        }

        with open(METADATA_DIR / f"{doc_id}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        written += 1

    print(f"Wrote {written} metadata JSON files to {METADATA_DIR}")


def merge_all() -> None:
    """Main merge: load triage results, update inventory, write metadata JSONs."""
    triage_data = load_triage_results()
    print(f"Total triage entries: {len(triage_data)}")

    # Read current inventory rows for metadata generation
    with open(INVENTORY_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    update_inventory(triage_data)
    write_metadata_jsons(triage_data, rows)

    # Summary stats
    include_counts = {}
    for entry in triage_data.values():
        inc = entry.get("include", "unknown")
        include_counts[inc] = include_counts.get(inc, 0) + 1
    print(f"Include breakdown: {include_counts}")


if __name__ == "__main__":
    merge_all()
