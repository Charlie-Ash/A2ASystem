# Builds the orchestrator's LangGraph graph
# run_orchestrator() used to run as a flat Python function (decide tool ->
# run tool -> generate reply -> update memory), now expressed as named nodes
# connected by edges.
from typing import TYPE_CHECKING

from langgraph.graph import StateGraph, START, END

from orchestrator.state import OrchestratorState

# OrchestratorLLM/ToolRouter are only used here as type hints. Importing them
# for real (rather than just for type-checking) would force this module to
# also import vllm transitively, even though build_graph itself never touches vllm
# directly and is perfectly happy running against fake/mocked stand-ins (see
# tests/fake_dependencies.py) that only need to match the same shape.
if TYPE_CHECKING:
    from orchestrator.llm import OrchestratorLLM
    from orchestrator.tool_router import ToolRouter


def build_graph(llm: "OrchestratorLLM", tool_router: "ToolRouter"):
    """Assemble and compile the orchestrator graph.

    Takes the LLM and tool router as arguments rather than constructing them
    itself, so the graph's wiring can be tested with fake/mocked versions of
    both without needing a GPU (see tests/test_graph_flow.py).
    """

    graph = StateGraph(OrchestratorState)

    # Node 1: ask the LLM which tool to use, then apply the two existing
    # rule-based safety nets on top of its answer (RAG always gets the user's
    # exact question; a note is never saved with blank content). These stay
    # inline here rather than as their own node because they're cheap,
    # deterministic follow-up to this same LLM call, not an independent step.
    def decide_tool(state: OrchestratorState) -> dict:

        tool_call = llm.tool_decision(state["user_message"])

        print("LLM tool decision: ", tool_call.tool)
        print("LLM tool argument: ", tool_call.args)

        if tool_call.tool == "rag":
            tool_call.args["query"] = state["user_message"]

        if tool_call.tool == "note" and not tool_call.args.get("content"):
            tool_call.args["content"] = state["user_message"]

        return {"tool_call": tool_call}

    # Routing function for the conditional edge below: reads the tool_call
    # that decide_tool just produced and picks which branch runs next. Its
    # return value must be one of the keys in the path_map passed to
    # add_conditional_edges further down.
    def route_to_tool(state: OrchestratorState) -> str:

        return state["tool_call"].tool  # "default" | "rag" | "note"

    # Builds one tool-node function per registered tool name. Each node is a
    # one-line delegation to that tool's own run() -- the actual tool
    # instances and their construction stay owned by ToolRouter, never
    # duplicated here.
    def make_tool_node(tool_name: str):

        def run_tool(state: OrchestratorState) -> dict:
            tool_result = tool_router.tools[tool_name].run(state["tool_call"].args)
            return {"tool_result": tool_result}

        return run_tool

    # Node: turn the tool's result into the natural-language reply the user
    # actually sees.
    def generate_response(state: OrchestratorState) -> dict:

        final_response = llm.generate_response(
            state["user_message"], state["tool_call"], state["tool_result"]
        )
        return {"final_response": final_response}

    # Node: summarize this turn into conversation memory. Returns no state
    # updates of its own (it only has a side effect: appending to
    # system_memory.md) -- an empty dict is a valid, no-op node return.
    def update_memory(state: OrchestratorState) -> dict:

        llm.orchestrator_mem_update(
            state["user_message"], state["tool_call"], state["tool_result"], state["final_response"]
        )
        return {}

    graph.add_node("decide_tool", decide_tool)
    graph.add_node("run_default_tool", make_tool_node("default"))
    graph.add_node("run_rag_tool", make_tool_node("rag"))
    graph.add_node("run_note_tool", make_tool_node("note"))
    graph.add_node("generate_response", generate_response)
    graph.add_node("update_memory", update_memory)

    graph.add_edge(START, "decide_tool")

    # Conditional edge: after decide_tool, route_to_tool's return value picks
    # which of these three nodes runs next -- this is the graph's one branch
    # point, standing in for what ToolRouter's dict lookup used to do inline.
    graph.add_conditional_edges(
        "decide_tool",
        route_to_tool,
        {
            "default": "run_default_tool",
            "rag": "run_rag_tool",
            "note": "run_note_tool",
        },
    )

    # All three branches rejoin at the same next step.
    graph.add_edge("run_default_tool", "generate_response")
    graph.add_edge("run_rag_tool", "generate_response")
    graph.add_edge("run_note_tool", "generate_response")

    graph.add_edge("generate_response", "update_memory")
    graph.add_edge("update_memory", END)

    return graph.compile()
