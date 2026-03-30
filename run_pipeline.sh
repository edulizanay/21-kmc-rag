#!/bin/bash
# ABOUTME: Overnight runner script — chains Phase 3 (chunking) → Phase 4 (embedding).
# ABOUTME: Designed to run via nohup and survive session disconnects.

set -e

LOG="pipeline_log.txt"
cd "$(dirname "$0")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"
}

log "=== Pipeline started ==="

# Activate venv
source .venv/bin/activate

# --- Phase 3: Chunking + Enrichment ---
log "Phase 3: Starting chunking + enrichment (concurrency=50)..."
if python3 -u -m src.chunking 50 2>&1 | tee -a "$LOG"; then
    log "Phase 3: DONE"
else
    EXIT_CODE=$?
    log "Phase 3: FAILED (exit code $EXIT_CODE)"
    log "Check if OpenRouter key limit was hit. Stopping pipeline."
    log "=== Pipeline stopped (Phase 3 failed) ==="
    exit 1
fi

# Verify chunks.json exists
if [ ! -f "data/chunks.json" ]; then
    log "ERROR: data/chunks.json not found after Phase 3. Stopping."
    exit 1
fi

CHUNK_COUNT=$(python3 -c "import json; print(len(json.load(open('data/chunks.json'))))")
log "Phase 3 produced $CHUNK_COUNT chunks"

# --- Phase 4A: Embedding + Vector Store ---
log "Phase 4A: Starting embedding + ChromaDB + BM25 + hybrid retrieval..."
if python3 -u -m src.vectorstore 2>&1 | tee -a "$LOG"; then
    log "Phase 4A: DONE"
else
    EXIT_CODE=$?
    log "Phase 4A: FAILED (exit code $EXIT_CODE)"
    log "=== Pipeline stopped (Phase 4A failed) ==="
    exit 1
fi

log "=== Pipeline complete! Phase 3 + 4A done. ==="
log "Next: Phase 4B (test set + tuning) and Phase 5 (LangGraph agent) — need Edu's input."
