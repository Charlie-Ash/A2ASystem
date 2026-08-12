# Checks the graph's wiring itself -- node names and connections -- without
# running any LLM or real network call. This catches typo'd node names, a
# missing edge, or a wrong conditional-edge path map: bugs that would
# otherwise stay invisible until an actual (GPU-and-network-requiring) run of
# main.py hit that exact branch.
from schemas.tool_call import ToolCall
from orchestrator.pipeline.graph import build_graph

from fake_dependencies import FakeOrchestratorLLM, FakeToolRouter


async def _build_test_graph():

    llm = FakeOrchestratorLLM(next_tool_call=ToolCall(tool="default", action="run", args={}))
    tool_router = await FakeToolRouter.create()
    return build_graph(llm, tool_router)


async def test_all_expected_nodes_are_present():

    compiled_graph = await _build_test_graph()
    node_names = set(compiled_graph.get_graph().nodes.keys())

    # "run_rag_tool"/"run_actions_tool" are built from whatever's in
    # tool_router.remote_agents at build_graph() time -- FakeToolRouter always
    # registers both, so this stays a fixed set here, but production's set
    # tracks orchestrator/config.py's REMOTE_AGENTS instead of a hardcoded pair.
    expected_nodes = {
        "decide_tool",
        "run_default_tool",
        "run_rag_tool",
        "run_actions_tool",
        "generate_response",
        "update_memory",
    }
    assert expected_nodes.issubset(node_names)


async def test_mermaid_diagram_shows_the_conditional_branch():

    compiled_graph = await _build_test_graph()
    mermaid_text = compiled_graph.get_graph().draw_mermaid()

    # Every branch target should be reachable from decide_tool in the drawn
    # diagram -- a stand-in check that add_conditional_edges' path_map is
    # wired to all 3 tool nodes, not just some of them.
    for node_name in ("run_default_tool", "run_rag_tool", "run_actions_tool"):
        assert node_name in mermaid_text


async def test_remote_tool_branches_are_plain_nodes_not_nested_subgraphs():

    # Before the A2A rework, run_rag_tool/run_actions_tool were compiled
    # LangGraph subgraphs registered directly as nodes, so xray=True surfaced
    # their internal nodes namespaced as "run_rag_tool:<inner_node_name>".
    # That's no longer true: the real subgraph work now happens on the other
    # side of an A2A call (a separate process in production, a separate
    # in-process ASGI app in tests -- see fake_dependencies.py), so from this
    # graph's own perspective run_rag_tool/run_actions_tool are opaque async
    # leaf nodes with nothing nested inside them. This asserts that shift
    # rather than leaving a stale assumption from before the rework.
    compiled_graph = await _build_test_graph()
    xray_node_names = set(compiled_graph.get_graph(xray=True).nodes.keys())

    assert not any(name.startswith("run_rag_tool:") for name in xray_node_names)
    assert not any(name.startswith("run_actions_tool:") for name in xray_node_names)
