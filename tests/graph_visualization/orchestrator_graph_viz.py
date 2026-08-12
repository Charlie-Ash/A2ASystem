# Renders the orchestrator's full graph to a PNG, without needing a GPU or
# real vllm/OrchestratorLLM objects, or a real network.
#
# Run directly: python tests/graph_visualization/orchestrator_graph_viz.py
# Deliberately NOT named test_*.py -- pytest's default discovery only picks up
# files matching that pattern, so a plain `pytest` run never executes this.
# It's meant to be run manually, on demand, whenever you want to eyeball the
# graph's current shape.
#
# Since the A2A rework (orchestrator/pipeline/graph.py's run_rag_tool/run_actions_tool
# are now real network calls, not nested LangGraph subgraphs -- see
# orchestrator/agents/remote_agent.py), there's no more internal structure to expand
# into via xray=True: RAG's/Actions' own extract->generate->... nodes now
# live inside their own standalone agent processes, invisible from the
# orchestrator's own graph. So this only draws the orchestrator's top-level
# shape now, not a merged diagram of all three graphs.
import asyncio
import sys
from pathlib import Path

# `pytest.ini`'s `pythonpath = src` setting only applies when pytest is the
# thing running the code. This script is run as a plain `python file.py`
# instead, so it adds both `src/` (for orchestrator.*, tools.*, schemas.*) and
# `tests/` (for fake_dependencies, which isn't a package -- it's imported as
# a bare top-level module the same way the pytest test files do) to sys.path
# itself.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from schemas.tool_call import ToolCall
from orchestrator.pipeline.graph import build_graph
from fake_dependencies import FakeOrchestratorLLM, FakeToolRouter

OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "orchestrator_graph.png"


async def main():
    # build_graph() is written to accept ANY object with the right shape for
    # llm/tool_router -- it never imports vllm itself (see the
    # `if TYPE_CHECKING:` guard at the top of orchestrator/pipeline/graph.py), so
    # duck-typed stand-ins work fine for drawing the graph's structure.
    llm = FakeOrchestratorLLM(next_tool_call=ToolCall(tool="default", action="run", args={}))

    # FakeToolRouter.create() spins up real, in-process A2AFastAPIApplication
    # instances (reached over httpx.ASGITransport, not real sockets) so that
    # tool_router.remote_agents ends up populated the same way it would be in
    # production -- with real RemoteAgent/AgentCard objects, just talking to
    # a fake subgraph underneath. build_graph() only needs remote_agents'
    # keys to decide how many branches to draw, so this is enough to produce
    # an accurate diagram of the real branch count/shape.
    tool_router = await FakeToolRouter.create()

    # No checkpointer passed -> build_graph() defaults to a fresh MemorySaver,
    # same as every real call site (orchestrator.py never passes one either).
    compiled_graph = build_graph(llm, tool_router)

    png_bytes = compiled_graph.get_graph(xray=True).draw_mermaid_png()

    # Save to disk so the diagram can be viewed by opening the file, since a
    # plain script run (no Jupyter/IPython kernel attached) can't show images
    # inline in a terminal.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(png_bytes)
    print(f"Saved orchestrator graph diagram to {OUTPUT_PATH}")

    # Standard LangGraph tutorial pattern for showing a graph image inline.
    # Only actually renders something inside a real IPython/Jupyter kernel
    # (e.g. a VS Code notebook cell) -- as a plain script run like this,
    # display() has nothing to draw into and silently does nothing, which is
    # why the PNG is also saved above.
    from IPython.display import Image, display
    display(Image(png_bytes))


if __name__ == "__main__":
    asyncio.run(main())
