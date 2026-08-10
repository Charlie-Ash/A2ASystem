# Stand-ins for OrchestratorLLM and the 3 real tools, used by the graph tests
# so they can run without a GPU. The real RAGTool/ActionsTool each load a
# whole vLLM engine (RAGTool a vector index too) in their __init__, so they
# must never be constructed here -- these fakes only need to satisfy the same
# shapes graph.py calls against (OrchestratorLLM's phase methods, and the
# Tool protocol's run()).
from langgraph.graph import StateGraph, START, END

from schemas.tool_call import ToolCall
from tools.ragTool.rag.state import RAGSubgraphState
from tools.actionsTool.actions.state import ActionsSubgraphState


class FakeOrchestratorLLM:
    """Records every call it receives and returns pre-set, canned answers.

    tool_decision's answer is configurable per test (via `next_tool_call`)
    since which tool gets picked is exactly what test_graph_flow.py wants to
    drive and check.
    """

    def __init__(self, next_tool_call: ToolCall):

        self.next_tool_call = next_tool_call
        self.tool_decision_calls = []
        self.generate_response_calls = []
        self.orchestrator_mem_update_calls = []

    def tool_decision(self, user_message, history_messages) -> ToolCall:

        self.tool_decision_calls.append((user_message, history_messages))
        return self.next_tool_call

    def generate_response(self, user_message, history_messages, tool_call, tool_result) -> str:

        self.generate_response_calls.append((user_message, history_messages, tool_call, tool_result))
        return f"canned reply about {tool_result['output']}"

    def orchestrator_mem_update(self, user_message, tool_call, tool_result, final_response) -> None:

        self.orchestrator_mem_update_calls.append(
            (user_message, tool_call, tool_result, final_response)
        )


class FakeTool:
    """A minimal stand-in satisfying tools.base.Tool: records the args it was
    called with and returns a fixed ToolResult."""

    def __init__(self, name: str, relay_verbatim: bool = False):

        self.name = name
        self.relay_verbatim = relay_verbatim
        self.received_args = None

    def run(self, tool_args: dict):

        self.received_args = tool_args
        return {"output": f"{self.name}-output", "relay_verbatim": self.relay_verbatim}


# Stands in for tools.ragTool.rag.graph.build_rag_subgraph()'s real output: an
# actually-compiled StateGraph over the real RAGSubgraphState (not a bare
# function), so tests exercise the real subgraph-as-node mechanics (shared
# tool_call/tool_result keys, checkpointer inheritance, xray namespacing)
# without ever touching vLLM/Qdrant/llama-index. Compiled with no explicit
# checkpointer, same as the real build_rag_subgraph(), so it inherits
# whatever checkpointer the parent test graph is built with.
def build_fake_rag_subgraph(canned_output="rag-output", relay_verbatim=True, received_args_sink=None):

    def fake_rag_run(state: RAGSubgraphState) -> dict:
        if received_args_sink is not None:
            received_args_sink["args"] = state["tool_call"].args
        return {"tool_result": {"output": canned_output, "relay_verbatim": relay_verbatim}}

    sub = StateGraph(RAGSubgraphState)
    sub.add_node("fake_rag_run", fake_rag_run)
    sub.add_edge(START, "fake_rag_run")
    sub.add_edge("fake_rag_run", END)
    return sub.compile()


# Stands in for tools.actionsTool.actions.graph.build_actions_subgraph()'s
# real output: an actually-compiled StateGraph over the real
# ActionsSubgraphState (not a bare function), same reasoning as
# build_fake_rag_subgraph above. received_messages_sink additionally records
# state["messages"] as received by this node, so tests can confirm the
# subgraph actually gets chat history passed in (see ActionsSubgraphState's
# "messages" field -- unlike RAGSubgraphState, it's a shared key).
def build_fake_actions_subgraph(
    canned_output="actions-output", relay_verbatim=False, received_args_sink=None, received_messages_sink=None
):

    def fake_actions_run(state: ActionsSubgraphState) -> dict:
        if received_args_sink is not None:
            received_args_sink["args"] = state["tool_call"].args
        if received_messages_sink is not None:
            received_messages_sink["messages"] = state.get("messages", [])
        return {"tool_result": {"output": canned_output, "relay_verbatim": relay_verbatim}}

    sub = StateGraph(ActionsSubgraphState)
    sub.add_node("fake_actions_run", fake_actions_run)
    sub.add_edge(START, "fake_actions_run")
    sub.add_edge("fake_actions_run", END)
    return sub.compile()


class FakeToolRouter:
    """Same shape as the real ToolRouter (a `.tools` dict keyed by tool
    name, plus a `.rag_subgraph`/`.actions_subgraph`), but built from
    FakeTool/fake-subgraph instances instead of the real, GPU/filesystem-
    touching tool classes."""

    def __init__(self):

        self.tools = {
            "default": FakeTool("default"),
        }

        # RAGTool/ActionsTool no longer satisfy the Tool protocol -- see tool_router.py.
        self.rag_received_args = {"args": None}
        self.rag_subgraph = build_fake_rag_subgraph(
            canned_output="rag-output", relay_verbatim=True, received_args_sink=self.rag_received_args
        )

        self.actions_received_args = {"args": None}
        self.actions_received_messages = {"messages": None}
        self.actions_subgraph = build_fake_actions_subgraph(
            canned_output="actions-output",
            relay_verbatim=False,
            received_args_sink=self.actions_received_args,
            received_messages_sink=self.actions_received_messages,
        )
