# Builds the orchestrator's LangGraph graph
# run_orchestrator() used to run as a flat Python function (decide tool ->
# run tool -> generate reply -> update memory), now expressed as named nodes
# connected by edges.
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from orchestrator.agents.remote_agent import call_remote_agent
from orchestrator.pipeline.state import OrchestratorState

# OrchestratorLLM/ToolRouter are only used here as type hints. Importing them
# for real (rather than just for type-checking) would force this module to
# also import vllm transitively, even though build_graph itself never touches vllm
# directly and is perfectly happy running against fake/mocked stand-ins (see
# tests/fake_dependencies.py) that only need to match the same shape.
if TYPE_CHECKING:
    from orchestrator.pipeline.llm import OrchestratorLLM
    from orchestrator.agents.tool_router import ToolRouter


def build_graph(llm: "OrchestratorLLM", tool_router: "ToolRouter", checkpointer=None):
    # build_graph's doc string below
    """Assemble and compile the orchestrator graph.

    Takes the LLM and tool router as arguments rather than constructing them
    itself, so the graph's wiring can be tested with fake/mocked versions of
    both without needing a GPU (see tests/test_graph_flow.py). `checkpointer`
    defaults to a fresh MemorySaver -- an in-RAM store, keyed by the
    `thread_id` passed into invoke()'s config, that snapshots this graph's
    state after every node runs. That's what lets state["messages"] (below)
    actually carry over from one invoke() call to the next instead of every
    turn starting from a blank state. Accepting it as a parameter (rather
    than always constructing one internally) keeps it swappable in tests.

    tool_router.remote_agents must already be populated (via
    `await tool_router.discover()`) before this is called -- the remote-tool
    branches are wired up from whatever's in that dict at build time, since
    LangGraph's graph shape is fixed once compiled.
    """

    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = StateGraph(OrchestratorState)

    # Node 1: ask the LLM which tool to use and record it. No more per-tool
    # arg-injection rules here (there used to be one for "rag" and one for
    # "actions") -- now that RAG/Actions are real A2A calls, the plain user
    # message text *is* the whole interface (the remote executors read it
    # straight off the A2A message, ignoring tool_call.args entirely -- see
    # run_remote_tool below), so there's nothing left for those rules to do.
    def decide_tool(state: OrchestratorState) -> dict:

        # History *before* this turn's message -- the new HumanMessage is
        # added below via the returned "messages" update, not read back by
        # this same call.
        history_messages = state.get("messages", [])

        tool_call = llm.tool_decision(
            state["user_message"], history_messages, tool_router.describe_tools_for_prompt()
        )

        print("LLM tool decision: ", tool_call.tool)
        print("LLM tool argument: ", tool_call.args)

        # Appended (not replacing) onto state["messages"] via the
        # add_messages reducer declared in state.py -- visible to
        # generate_response later in this same run, and to decide_tool on
        # the next turn once MemorySaver has checkpointed it.
        return {"tool_call": tool_call, "messages": [HumanMessage(content=state["user_message"])]}

    # Routing function for the conditional edge below: reads the tool_call
    # that decide_tool just produced and picks which branch runs next. Its
    # return value must be one of the keys in the path_map built below --
    # "default" plus however many remote agents were actually discovered.
    def route_to_tool(state: OrchestratorState) -> str:

        return state["tool_call"].tool

    # Builds a node for a local, in-process Tool-protocol instance (today,
    # only "default"). One-line delegation to that tool's own run() -- the
    # actual tool instances and their construction stay owned by ToolRouter,
    # never duplicated here.
    def make_local_tool_node(tool_name: str):

        def run_local_tool(state: OrchestratorState) -> dict:
            tool_result = tool_router.tools[tool_name].run(state["tool_call"].args)
            return {"tool_result": tool_result}

        return run_local_tool

    # Builds a node for a remote A2A agent: sends this turn's user message
    # over the network (via call_remote_agent, see orchestrator/agents/remote_agent.py)
    # instead of calling an in-process subgraph. This is an async node --
    # LangGraph runs sync and async nodes side-by-side fine, but the graph
    # as a whole must be invoked via .ainvoke()/.astream() rather than
    # .invoke() once any node is async (see orchestrator.py).
    def make_remote_tool_node(tool_name: str):

        async def run_remote_tool(state: OrchestratorState) -> dict:
            agent = tool_router.remote_agents[tool_name]
            tool_result = await call_remote_agent(agent, state["user_message"])
            return {"tool_result": tool_result}

        return run_remote_tool

    graph.add_node("decide_tool", decide_tool)
    graph.add_node("run_default_tool", make_local_tool_node("default"))
    graph.add_edge("run_default_tool", "generate_response")

    # One node/edge pair per discovered remote agent, instead of a fixed
    # "run_rag_tool"/"run_actions_tool" pair -- however many entries
    # tool_router.remote_agents ends up with (per orchestrator/config.py's
    # REMOTE_AGENTS), that's how many branches this graph gets. This is the
    # actual fix for ToolRouter/graph.py no longer being allowed to look like
    # "3 hardcoded local function names" once two of them are network calls.
    path_map = {"default": "run_default_tool"}

    for agent_name in tool_router.remote_agents:

        node_name = f"run_{agent_name}_tool"
        graph.add_node(node_name, make_remote_tool_node(agent_name))
        graph.add_edge(node_name, "generate_response")
        path_map[agent_name] = node_name

    # Node: turn the tool's result into the natural-language reply the user
    # actually sees.
    def generate_response(state: OrchestratorState) -> dict:

        # state["messages"] already includes this turn's HumanMessage
        # (decide_tool's update was merged in before this node ran) --
        # drop it so history_messages keeps the same "turns before this
        # one" contract build_response_prompt/build_tool_decision_prompt
        # both expect, with user_message supplied separately.
        history_messages = state["messages"][:-1]
        final_response = llm.generate_response(
            state["user_message"], history_messages, state["tool_call"], state["tool_result"]
        )
        return {"final_response": final_response, "messages": [AIMessage(content=final_response)]}

    # Node: summarize this turn into the chat log. Returns no state updates
    # of its own (it only has a side effect: appending to chat_log.md) --
    # an empty dict is a valid, no-op node return.
    def update_memory(state: OrchestratorState) -> dict:

        llm.orchestrator_mem_update(
            state["user_message"], state["tool_call"], state["tool_result"], state["final_response"]
        )
        return {}

    graph.add_node("generate_response", generate_response)
    graph.add_node("update_memory", update_memory)

    graph.add_edge(START, "decide_tool")

    # Conditional edge: after decide_tool, route_to_tool's return value picks
    # which branch runs next -- this is the graph's one branch point,
    # standing in for what ToolRouter's dict lookup used to do inline.
    graph.add_conditional_edges("decide_tool", route_to_tool, path_map)

    graph.add_edge("generate_response", "update_memory")
    graph.add_edge("update_memory", END)

    return graph.compile(checkpointer=checkpointer)
