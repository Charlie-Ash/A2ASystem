# Renders the orchestrator's full graph (decide_tool's 3-way branch, with the
# RAG branch expanded to its real internal nodes) to a PNG, without needing a
# GPU or real vllm/llama_index/OrchestratorLLM objects.
#
# Run directly: python tests/graph_visualization/orchestrator_graph_viz.py
# Deliberately NOT named test_*.py -- pytest's default discovery only picks up
# files matching that pattern, so a plain `pytest` run never executes this.
# It's meant to be run manually, on demand, whenever you want to eyeball the
# graph's current shape.
import sys
from pathlib import Path
from types import SimpleNamespace

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
from orchestrator.graph import build_graph
from tools.ragTool.rag.graph import build_rag_subgraph
from tools.actionsTool.actions.graph import build_actions_subgraph
from fake_dependencies import FakeOrchestratorLLM, FakeTool

OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "orchestrator_graph.png"


def main():
    # build_graph() is written to accept ANY object with the right shape for
    # llm/tool_router -- it never imports vllm/llama_index itself (see the
    # `if TYPE_CHECKING:` guard at the top of orchestrator/graph.py), so
    # duck-typed stand-ins work fine for drawing the graph's structure.
    # FakeOrchestratorLLM (from tests/fake_dependencies.py, the same fakes
    # the real pytest suite uses) just needs a ToolCall to construct; that
    # value is never actually read for graph structure/drawing purposes.
    llm = FakeOrchestratorLLM(next_tool_call=ToolCall(tool="default", action="run", args={}))

    # tool_router only needs to look like a ToolRouter: a `.tools` dict
    # (for the default tool node) and `.rag_subgraph`/`.actions_subgraph`
    # attributes (the compiled subgraphs that get registered directly as the
    # "run_rag_tool"/"run_actions_tool" nodes). SimpleNamespace is a plain
    # "bag of attributes" object -- an easy way to satisfy that shape without
    # writing a whole class for it.
    #
    # Using the REAL compiled RAG/Actions subgraphs here (instead of
    # FakeToolRouter's rag_subgraph/actions_subgraph, which pytest uses and
    # which collapse each into one fake stub node) means get_graph(xray=True)
    # below can expand both branches into their true internal nodes.
    real_rag_subgraph = build_rag_subgraph(index=None, llm=None, sampling_params=None)
    real_actions_subgraph = build_actions_subgraph(llm=None, sampling_params=None)
    tool_router = SimpleNamespace(
        tools={"default": FakeTool("default")},
        rag_subgraph=real_rag_subgraph,
        actions_subgraph=real_actions_subgraph,
    )

    # No checkpointer passed -> build_graph() defaults to a fresh MemorySaver,
    # same as every real call site (orchestrator.py never passes one either).
    compiled_graph = build_graph(llm, tool_router)

    # xray=True tells LangGraph to "look inside" any node that is itself a
    # compiled subgraph (here, run_rag_tool/run_actions_tool) and draw its
    # internal nodes too, namespaced as "run_rag_tool:extract_query" etc.,
    # instead of drawing it as one opaque box.
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
    main()
