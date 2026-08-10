# Shared constants for the Actions tool.

import os
from pathlib import Path

# Resolved relative to this file rather than the process's CWD (same pattern
# as ragTool/config.py).
ACTIONSTOOL_DIR = Path(__file__).resolve().parent
AGENTSYSTEM_ROOT = ACTIONSTOOL_DIR.parent.parent.parent

# Notes are written into the project's shared top-level data/ folder, in
# their own subfolder (moved verbatim from the old notesTool/config.py).
NOTES_DIR = AGENTSYSTEM_ROOT / "data" / "notes"

# Small (1B) instruct model -- this agent's job (write a short note from a
# user request, optionally using recent chat history) doesn't need anywhere
# near the capacity RAG's or the orchestrator's models have, and there's very
# little VRAM budget left to spend: on a 32 GiB card, the orchestrator's
# gemma-4-E4B-it (0.6) and RAG's gemma-4-E2B-it-qat-w4a16-ct (0.3) already
# commit 0.9 of the device when both load in the same process (main.py,
# since RAG is still called in-process today). That leaves ~0.1 (~3.2 GiB)
# for this engine -- enough for gemma-3-1b-it's ~2 GiB bf16 weights plus
# headroom for KV cache/activations, but not much more.
LLM_MODEL = os.environ.get("ACTIONS_LLM_MODEL", "google/gemma-3-1b-it")
GPU_MEMORY_UTILIZATION = float(os.environ.get("ACTIONS_GPU_MEMORY_UTILIZATION", "0.1"))
# Note content generation is short -- no need for the 4096 the other two
# engines use.
MAX_MODEL_LEN = int(os.environ.get("ACTIONS_MAX_MODEL_LEN", "2048"))

# Standalone A2A server bind address (see a2a/a2a_server.py). Port follows
# RAG's 8001.
ACTIONS_A2A_HOST = os.environ.get("ACTIONS_A2A_HOST", "0.0.0.0")
ACTIONS_A2A_PORT = int(os.environ.get("ACTIONS_A2A_PORT", "8002"))
