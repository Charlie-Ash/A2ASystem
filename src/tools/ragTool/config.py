# Shared constants for the RAG tool.

import os
from pathlib import Path

# Resolved relative to this file rather than the process's CWD.
RAGTOOL_DIR = Path(__file__).resolve().parent
AGENTSYSTEM_ROOT = RAGTOOL_DIR.parent.parent.parent

# Qdrant's on-disk store lives inside this tool's own folder.
QDRANT_DB_PATH = str(RAGTOOL_DIR / "qdrant_db")

# Documents to ingest live in the project's shared top-level data/ folder.
DATA_DIR = AGENTSYSTEM_ROOT / "data"

COLLECTION_NAME = "my_docs"

EMBEDDING_MODEL = "BAAI/bge-m3"

# Small, official Google QAT-quantized checkpoint (~7.3 GiB VRAM). A
# different size class from the orchestrator's E4B-it model, so the two
# can coexist on one GPU without competing for memory.
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "google/gemma-4-E2B-it-qat-w4a16-ct")
# Lowered to leave room for the orchestrator's larger gemma-4-E4B-it engine
# (~15.28 GiB weights alone), which loads on the same GPU. On a 32 GiB card,
# 0.3 gives ~9.6 GiB (this model's ~7.3 GiB weights + ~2.3 GiB headroom).
GPU_MEMORY_UTILIZATION = float(os.environ.get("RAG_GPU_MEMORY_UTILIZATION", "0.3"))
MAX_MODEL_LEN = int(os.environ.get("RAG_MAX_MODEL_LEN", "4096"))

# Standalone A2A server bind address (see a2a_server.py). 
RAG_A2A_HOST = os.environ.get("RAG_A2A_HOST", "0.0.0.0")
RAG_A2A_PORT = int(os.environ.get("RAG_A2A_PORT", "8001"))
