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
# near the capacity RAG's or the orchestrator's models have. This engine runs
# in its own standalone A2A server process (see a2a/a2a_server.py); it is not
# sharing a process with the orchestrator or RAG. In practice this project
# runs Actions in its own serialized session (Actions + orchestrator only,
# RAG's server stopped) rather than concurrently with both other engines --
# see A2A_DIAGNOSIS.md, which found the default 0.1 (~3.2 GiB) too tight even
# in complete isolation once fixed per-process overhead (CUDA context,
# torch.compile cache, CUDA graphs) is accounted for, and
# A2A_DIAGNOSIS_FIX_PLAN.md for the measured numbers and current default.
LLM_MODEL = os.environ.get("ACTIONS_LLM_MODEL", "google/gemma-3-1b-it")
GPU_MEMORY_UTILIZATION = float(os.environ.get("ACTIONS_GPU_MEMORY_UTILIZATION", "0.1"))
# Note content generation is short -- no need for the 4096 the other two
# engines use.
MAX_MODEL_LEN = int(os.environ.get("ACTIONS_MAX_MODEL_LEN", "2048"))

# Standalone A2A server bind address (see a2a/a2a_server.py). Port follows
# RAG's 8001.
ACTIONS_A2A_HOST = os.environ.get("ACTIONS_A2A_HOST", "0.0.0.0")
ACTIONS_A2A_PORT = int(os.environ.get("ACTIONS_A2A_PORT", "8002"))
