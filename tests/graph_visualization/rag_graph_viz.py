# Renders the RAG tool's subgraph (extract_query -> retrieve -> generate_answer)
# to a PNG, without needing a GPU or real vllm/llama_index objects.
#
# Run directly: python tests/graph_visualization/rag_graph_viz.py
# Deliberately NOT named test_*.py -- pytest's default discovery only picks up
# files matching that pattern, so a plain `pytest` run never executes this.
# It's meant to be run manually, on demand, whenever you want to eyeball the
# graph's current shape.
import sys
from pathlib import Path

# `pytest.ini`'s `pythonpath = src` setting only applies when pytest is the
# thing running the code. This script is run as a plain `python file.py`
# instead, so it has to add `src/` to sys.path itself, the same way any
# standalone script would, before it can do `from tools.ragTool.rag.graph import ...`.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tools.ragTool.rag.graph import build_rag_subgraph

OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "rag_graph.png"


def main():
    # build_rag_subgraph()'s index/llm/sampling_params params are only ever
    # read inside its node *functions* (retrieve, generate_answer) when the
    # graph is actually invoked/run. Building the graph and asking it to draw
    # itself never calls those node functions -- it just inspects the graph's
    # structure (nodes + edges) -- so passing None for all three is safe here.
    compiled_graph = build_rag_subgraph(index=None, llm=None, sampling_params=None)

    # get_graph() returns a langgraph "Graph" object describing the compiled
    # graph's nodes/edges. draw_mermaid_png() turns that into an actual PNG
    # image by sending the diagram's Mermaid text to the mermaid.ink web
    # service and getting a rendered image back -- this needs internet access.
    png_bytes = compiled_graph.get_graph().draw_mermaid_png()

    # Save to disk so the diagram can be viewed by opening the file, since a
    # plain script run (no Jupyter/IPython kernel attached) can't show images
    # inline in a terminal.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(png_bytes)
    print(f"Saved RAG subgraph diagram to {OUTPUT_PATH}")

    # This is the standard LangGraph tutorial pattern for showing a graph
    # image inline. It only actually renders something when code is running
    # inside a real IPython/Jupyter kernel (e.g. a VS Code notebook cell) --
    # run as a plain script like this, display() has nothing to draw into and
    # silently does nothing, which is why the PNG is also saved above.
    from IPython.display import Image, display
    display(Image(png_bytes))


if __name__ == "__main__":
    main()
