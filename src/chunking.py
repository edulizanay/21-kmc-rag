# ABOUTME: Phase 3 — splits processed documents into retrieval-ready chunks with metadata.
# ABOUTME: Produces data/chunks.json with enriched chunks ready for embedding.

import csv
import json

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.config import (
    CHUNKS_PATH,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    INVENTORY_PATH,
    METADATA_DIR,
    PROCESSED_TEXTS_PATH,
)


def load_metadata(doc_id: str) -> dict:
    """Load per-document metadata JSON."""
    meta_path = METADATA_DIR / f"{doc_id}.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def split_document(doc_id: str, text: str, file_type: str) -> list[dict]:
    """Split a document into chunks based on its type."""
    if file_type == "md":
        return _split_markdown(doc_id, text)
    else:
        return _split_recursive(doc_id, text)


def _build_header_chain(metadata: dict) -> str:
    """Reconstruct the markdown header hierarchy as a readable string."""
    parts = []
    for key in ("h1", "h2", "h3"):
        if key in metadata:
            parts.append(metadata[key])
    return " > ".join(parts)


def _split_markdown(doc_id: str, text: str) -> list[dict]:
    """Split markdown by headers first, then by size."""
    headers = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    md_chunks = md_splitter.split_text(text)

    # Further split large sections
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    final_chunks = []
    for i, chunk in enumerate(md_chunks):
        # Prepend header chain back into content (the splitter strips them)
        header_chain = _build_header_chain(chunk.metadata)
        content = (
            f"{header_chain}\n\n{chunk.page_content}"
            if header_chain
            else chunk.page_content
        )

        if len(content) > DEFAULT_CHUNK_SIZE:
            sub_chunks = text_splitter.split_text(content)
            for j, sub in enumerate(sub_chunks):
                final_chunks.append(
                    {
                        "page_content": sub,
                        "chunk_index": len(final_chunks),
                        "doc_id": doc_id,
                        "section": header_chain,
                    }
                )
        else:
            final_chunks.append(
                {
                    "page_content": content,
                    "chunk_index": len(final_chunks),
                    "doc_id": doc_id,
                    "section": header_chain,
                }
            )

    return final_chunks


def _split_recursive(doc_id: str, text: str) -> list[dict]:
    """Split by recursive character boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )
    texts = splitter.split_text(text)
    return [
        {"page_content": t, "chunk_index": i, "doc_id": doc_id}
        for i, t in enumerate(texts)
    ]


def enrich_chunk(chunk: dict, meta: dict) -> dict:
    """Add a contextual prefix to a chunk using document metadata and section headers."""
    doc_name = meta.get("filename", chunk["doc_id"])
    section = chunk.get("section", "")

    # Use section header if available, otherwise fall back to doc-level summary
    if section:
        prefix = f"From {doc_name}, section: {section}."
    else:
        summary = meta.get("summary", "")
        content_type = meta.get("content_type", "")
        audience = meta.get("audience", "")
        first_sentence = summary.split(".")[0].strip() + "." if summary else ""

        parts = [f"From {doc_name}"]
        if content_type or audience:
            details = ", ".join(filter(None, [content_type, audience]))
            parts[0] += f" ({details})"
        parts[0] += ":"
        if first_sentence:
            parts.append(first_sentence)
        prefix = " ".join(parts)

    chunk["page_content"] = f"{prefix}\n\n{chunk['page_content']}"
    return chunk


def chunk_all() -> list[dict]:
    """Split all processed documents into enriched chunks."""
    with open(PROCESSED_TEXTS_PATH, encoding="utf-8") as f:
        processed = json.load(f)

    # Load inventory for file types
    with open(INVENTORY_PATH, encoding="utf-8") as f:
        inventory = {r["doc_id"]: r for r in csv.DictReader(f)}

    print(f"Chunking {len(processed)} documents...", flush=True)

    all_chunks = []
    for doc_id, entry in processed.items():
        text = entry["text"]
        file_type = inventory.get(doc_id, {}).get("file_type", "txt")
        chunks = split_document(doc_id, text, file_type)
        all_chunks.extend(chunks)

    print(f"Total chunks after splitting: {len(all_chunks)}", flush=True)

    # Enrich with contextual prefixes from metadata
    print(f"Enriching {len(all_chunks)} chunks with metadata prefixes...", flush=True)
    for chunk in all_chunks:
        meta = load_metadata(chunk["doc_id"])
        enrich_chunk(chunk, meta)

    # Attach metadata and serialize
    output_chunks = []
    for chunk in all_chunks:
        doc_id = chunk["doc_id"]
        meta = load_metadata(doc_id)
        output_chunks.append(
            {
                "page_content": chunk["page_content"],
                "metadata": {
                    "doc_id": doc_id,
                    "doc_name": meta.get("filename", ""),
                    "summary": meta.get("summary", ""),
                    "topic_tags": meta.get("topic_tags", []),
                    "audience": meta.get("audience", ""),
                    "content_type": meta.get("content_type", ""),
                    "chunk_index": chunk["chunk_index"],
                },
            }
        )

    # Sort by doc_id and chunk_index
    output_chunks.sort(
        key=lambda c: (c["metadata"]["doc_id"], c["metadata"]["chunk_index"])
    )

    # Write chunks
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_chunks, f, ensure_ascii=False)
    print(f"Wrote {len(output_chunks)} chunks to {CHUNKS_PATH}", flush=True)

    # Update inventory with chunk counts
    chunk_counts = {}
    for c in output_chunks:
        did = c["metadata"]["doc_id"]
        chunk_counts[did] = chunk_counts.get(did, 0) + 1

    with open(INVENTORY_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    for row in rows:
        if row["doc_id"] in chunk_counts:
            row["chunking_complete"] = "yes"
            row["chunk_count"] = str(chunk_counts[row["doc_id"]])

    with open(INVENTORY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Updated inventory with chunk counts for {len(chunk_counts)} docs", flush=True
    )

    return output_chunks


if __name__ == "__main__":
    chunk_all()
