# ABOUTME: Phase 1 — walks the KMC corpus folder, catalogs all files into inventory.csv.
# ABOUTME: Produces the master file list that drives all downstream processing.

import csv
import os
from pathlib import Path

from src.config import CORPUS_DIR, DATA_DIR, INVENTORY_PATH, SKIP_EXTENSIONS


def walk_corpus() -> list[dict]:
    """Recursively walk the corpus folder and catalog every file."""
    files = []
    doc_counter = 0

    for root, _dirs, filenames in os.walk(CORPUS_DIR):
        for filename in sorted(filenames):
            filepath = Path(root) / filename
            ext = filepath.suffix.lower()

            if ext in SKIP_EXTENSIONS:
                continue

            # Skip hidden files and OS artifacts
            if filename.startswith(".") or filename == "Thumbs.db":
                continue

            doc_counter += 1
            doc_id = f"doc_{doc_counter:03d}"

            file_size = filepath.stat().st_size
            relative_folder = str(Path(root).relative_to(CORPUS_DIR))

            files.append(
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "file_type": ext.lstrip("."),
                    "file_size": file_size,
                    "char_count": "",
                    "path": str(filepath),
                    "folder": relative_folder,
                    "status": "unprocessed",
                    "include": "",
                    "content_type": "",
                    "sensitivity_flag": "",
                    "summary": "",
                    "topic_tags": "",
                    "audience": "",
                    "preprocessing_complete": "",
                    "extraction_quality": "",
                    "processed_char_count": "",
                    "chunking_complete": "",
                    "chunk_count": "",
                }
            )

    return files


def write_inventory(files: list[dict]) -> None:
    """Write the file catalog to inventory.csv."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "doc_id",
        "filename",
        "file_type",
        "file_size",
        "char_count",
        "path",
        "folder",
        "status",
        "include",
        "content_type",
        "sensitivity_flag",
        "summary",
        "topic_tags",
        "audience",
        "preprocessing_complete",
        "extraction_quality",
        "processed_char_count",
        "chunking_complete",
        "chunk_count",
    ]

    with open(INVENTORY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(files)


def build_inventory() -> list[dict]:
    """Walk corpus and write inventory. Returns the file list."""
    files = walk_corpus()
    write_inventory(files)
    return files


if __name__ == "__main__":
    files = build_inventory()
    print(f"Cataloged {len(files)} files to {INVENTORY_PATH}")

    # Show breakdown by type
    from collections import Counter

    type_counts = Counter(f["file_type"] for f in files)
    for ftype, count in type_counts.most_common():
        print(f"  {ftype}: {count}")
