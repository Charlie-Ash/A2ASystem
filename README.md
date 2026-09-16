# A2ASystem

A locally-hosted, multi-agent knowledge assistant built on **Google's Agent-to-Agent (A2A) protocol** and **LangGraph**, running entirely on a single consumer GPU via **vLLM**.

## About

A2ASystem is an orchestrator ("Secretary") that routes each user message to the right specialist agent using vLLM guided/structured decoding, rather than a hand-parsed prompt:

- **RAG Agent** — answers questions from a company's own documents (PDF ingestion → vector search → grounded answer).
- **Actions Agent** — turns a request into a saved note on disk.

Each agent is its own long-running process with its own vLLM engine, and the orchestrator talks to them over the network using the A2A protocol rather than calling them as local functions.

This repository is also the codebase behind an ongoing case-study research paper (submitted to **ICS2026**) on incrementally promoting a local multi-agent assistant onto a real networked agent protocol on constrained, single-GPU consumer hardware, and measuring what that promotion actually costs in latency, routing accuracy, and VRAM. See [Project evolution & branches](#project-evolution--branches) below.

## Architecture

Three independent processes share one GPU, each running its own vLLM engine:

| Process | Role | Entry point | Default address |
|---|---|---|---|
| **Orchestrator** | User-facing REPL; A2A **client**; decides which agent to call and composes the final reply | `src/main.py` | (local, stdin/stdout) |
| **RAG Agent** | A2A **server**; answers from ingested documents | `src/tools/ragTool/a2a/a2a_server.py` | `http://0.0.0.0:8001` |
| **Actions Agent** | A2A **server**; writes notes to disk | `src/tools/actionsTool/a2a/a2a_server.py` | `http://0.0.0.0:8002` |

Internally, every agent is its own compiled **LangGraph** graph:

- **Orchestrator** — `decide_tool` (guided-decoding tool routing) → route to the local `default` tool or a discovered remote agent → `generate_response` → `update_memory`.
- **RAG Agent** — `extract_query` → `retrieve` (Qdrant vector search) → `generate_answer`.
- **Actions Agent** — `extract_request` → `generate_content` (guided-decoding note content + filename) → `save_note`.

At startup, the orchestrator discovers each configured remote agent's `AgentCard` over HTTP; an agent that's unreachable is simply excluded from routing for that session rather than causing a crash. Since the Actions agent needs conversation context (e.g. "note down your previous answer") but lives in a different process, the orchestrator attaches a slice of recent chat history to the outgoing A2A message's metadata, and the remote agent reconstructs it on its side.

Conversation state is checkpointed across turns with LangGraph's `AsyncSqliteSaver` (`data/checkpoints/checkpoints.db`), so the orchestrator can restart without losing the current conversation. A separate, LLM-summarized memory log (`data/chat_log/chat_log.md`) is also kept per session.

## Project structure

```
A2ASystem/
├── README.md
├── pytest.ini                    # pythonpath=src, asyncio_mode=auto
├── requirements.txt
├── data/                         # runtime data (mostly generated at run time)
│   └── Test_PDF.pdf              # sample document for RAG ingestion
│   # created at runtime: chat_log/, checkpoints/, notes/
├── src/
│   ├── main.py                   # CLI entry point (Orchestrator REPL)
│   ├── chat_history.py           # shared LangChain-messages <-> wire-format helpers
│   ├── orchestrator/              # the Orchestrator agent (A2A client)
│   │   ├── orchestrator.py       # session/thread lifecycle, run_orchestrator()
│   │   ├── config.py             # REMOTE_AGENTS list, checkpoint DB path
│   │   ├── memory_manager.py     # chat_log.md + thread_id.txt file I/O
│   │   ├── agents/
│   │   │   ├── remote_agent.py   # A2A client: discover + call remote agents
│   │   │   └── tool_router.py    # local "default" tool + discovered remote agents
│   │   └── pipeline/
│   │       ├── graph.py          # orchestrator's LangGraph StateGraph
│   │       ├── state.py          # OrchestratorState
│   │       ├── llm.py            # OrchestratorLLM (vLLM wrapper)
│   │       └── prompts.py        # prompt builders
│   ├── schemas/
│   │   ├── tool_call.py          # ToolCall pydantic model
│   │   └── tool_schema.py        # guided-decoding schema builder
│   └── tools/
│       ├── base.py               # Tool protocol + ToolResult contract
│       ├── gpu_utils.py          # VRAM pre-flight check
│       ├── defaultTool/          # trivial in-process fallback tool
│       ├── ragTool/              # standalone RAG A2A agent (own process)
│       │   ├── config.py
│       │   ├── rag/              # dataIngestion, userQuery, graph, state, tool_class
│       │   └── a2a/              # a2a_server.py, agent_executor.py
│       └── actionsTool/          # standalone Actions A2A agent (own process)
│           ├── config.py
│           ├── actions/          # noteWriter, note_schema, graph, state, tool_class
│           └── a2a/              # a2a_server.py, agent_executor.py
└── tests/
    ├── fake_dependencies.py      # GPU-free, network-free fakes for testing
    ├── test_*.py                 # A2A servers, executors, graph structure/flow, checkpointer
    └── graph_visualization/      # standalone scripts (not pytest) to render graphs to PNG
```

## Project evolution & branches

This repository documents three incremental phases of the same system, each a deliberate stepping stone toward the next — the subject of the accompanying ICS2026 paper.

| Phase | Branch | Summary |
|---|---|---|
| **System I** | [`guided-decoding-system`](../../tree/guided-decoding-system) | The original hand-rolled, single-process orchestrator: vLLM guided/structured decoding drives tool routing directly, with a flat markdown file as memory. |
| **System II** | [`lang-graph-intergration-system`](../../tree/lang-graph-intergration-system) | The same control flow re-expressed as an explicit LangGraph `StateGraph` (supervisor pattern). RAG is promoted to a compiled subgraph and a checkpointer adds real graph-state chat history — still a single process. |
| **System III** | `main` (this branch) | RAG and a new Actions tool are promoted out-of-process into standalone agents speaking the A2A protocol. The orchestrator becomes a genuine A2A client with cross-process memory and persistent checkpointing. |

Each phase keeps the same underlying scenario (route a user request, answer from documents, or save a note) so the cost of each architectural step — latency, routing accuracy, GPU memory — can be measured and compared directly.

## Environment & requirements

- **Python**: developed against Python 3.14.
- **GPU**: an NVIDIA CUDA-capable GPU is required — every agent runs its own vLLM engine. Developed and evaluated on a single consumer GPU (RTX 5090, ~32 GiB VRAM). Running all three engines concurrently is VRAM-tight; each agent's `*_GPU_MEMORY_UTILIZATION` env var (see below) controls how much of the card it reserves, and `src/tools/gpu_utils.py` performs a pre-flight check before loading a model. If you're VRAM-constrained, don't run all three servers at once — start only the agents you need for a given session.
- **Models used by default**: orchestrator `google/gemma-4-E4B-it`, RAG `google/gemma-4-E2B-it-qat-w4a16-ct` (quantized), Actions `google/gemma-3-1b-it`.
- **Key dependencies** (see `requirements.txt`): `vllm`, `langgraph` + `langgraph-checkpoint-sqlite`, `langchain-core`, `a2a-sdk[http-server]`, `uvicorn`, `llama-index` + `qdrant-client` (RAG), `pymupdf`, `sentence-transformers`, `pytest` + `pytest-asyncio`.

## Setup

```bash
git clone https://github.com/Charlie-Ash/A2ASystem.git
cd A2ASystem
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux

pip install -r requirements.txt

# CUDA-enabled torch is installed separately (see requirements.txt):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Running the project

Start the two agent servers first, then the orchestrator — agent discovery happens once, at orchestrator startup, and an agent that isn't reachable yet is simply skipped for that session.

```bash
# Terminal 1 — RAG agent (http://0.0.0.0:8001)
cd src
python -m tools.ragTool.a2a.a2a_server

# Terminal 2 — Actions agent (http://0.0.0.0:8002)
cd src
python -m tools.actionsTool.a2a.a2a_server

# Terminal 3 — Orchestrator (REPL)
cd src
python main.py
```

Type `bye` in the orchestrator REPL to exit (you'll be asked whether to save a summarized memory of the session).

Useful environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `REMOTE_AGENTS` | `rag@http://localhost:8001,actions@http://localhost:8002` | Which A2A agents the orchestrator discovers, as `name@url` pairs |
| `REMOTE_AGENTS_VERBATIM` | `rag` | Which agents' successful results are relayed to the user as-is instead of composed into a reply |
| `CHECKPOINT_DB_PATH` | `data/checkpoints/checkpoints.db` | Where the persistent LangGraph checkpointer writes conversation state |
| `ORCHESTRATOR_GPU_MEMORY_UTILIZATION` | `0.6` | Fraction of VRAM reserved for the orchestrator's vLLM engine |
| `RAG_GPU_MEMORY_UTILIZATION` | `0.3` | Fraction of VRAM reserved for the RAG agent's vLLM engine |
| `ACTIONS_GPU_MEMORY_UTILIZATION` | `0.1` | Fraction of VRAM reserved for the Actions agent's vLLM engine |
| `RAG_A2A_HOST` / `RAG_A2A_PORT` | `0.0.0.0` / `8001` | RAG agent server bind address |
| `ACTIONS_A2A_HOST` / `ACTIONS_A2A_PORT` | `0.0.0.0` / `8002` | Actions agent server bind address |

## Testing

```bash
pytest
```

Tests run without a GPU or real network access, using fakes (`tests/fake_dependencies.py`) and in-process ASGI transports for the A2A servers.

Two standalone scripts (not part of the pytest suite — run directly) render each agent's compiled LangGraph to a PNG for inspection:

```bash
python tests/graph_visualization/orchestrator_graph_viz.py
python tests/graph_visualization/rag_graph_viz.py
```
