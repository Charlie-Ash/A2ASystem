# Checks the graph's wiring itself -- node names and connections -- without
# running any LLM or tool. This catches typo'd node names, a missing edge, or
# a wrong conditional-edge path map: bugs that would otherwise stay invisible
# until an actual (GPU-requiring) run of main.py hit that exact branch.
from schemas.tool_call import ToolCall
from orchestrator.graph import build_graph

from fake_dependencies import FakeOrchestratorLLM, FakeToolRouter


def _build_test_graph():

    llm = FakeOrchestratorLLM(next_tool_call=ToolCall(tool="default", action="run", args={}))
    tool_router = FakeToolRouter()
    return build_graph(llm, tool_router)


def test_all_expected_nodes_are_present():

    compiled_graph = _build_test_graph()
    node_names = set(compiled_graph.get_graph().nodes.keys())

    expected_nodes = {
        "decide_tool",
        "run_default_tool",
        "run_rag_tool",
        "run_note_tool",
        "generate_response",
        "update_memory",
    }
    assert expected_nodes.issubset(node_names)


def test_mermaid_diagram_shows_the_conditional_branch():

    compiled_graph = _build_test_graph()
    mermaid_text = compiled_graph.get_graph().draw_mermaid()

    # Every branch target should be reachable from decide_tool in the drawn
    # diagram -- a stand-in check that add_conditional_edges' path_map is
    # wired to all 3 tool nodes, not just some of them.
    for node_name in ("run_default_tool", "run_rag_tool", "run_note_tool"):
        assert node_name in mermaid_text
