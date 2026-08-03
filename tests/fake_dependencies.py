# Stand-ins for OrchestratorLLM and the 3 real tools, used by the graph tests
# so they can run without a GPU. The real RAGTool alone loads a whole vLLM
# engine + vector index in its __init__, so it must never be constructed
# here -- these fakes only need to satisfy the same shapes graph.py calls
# against (OrchestratorLLM's phase methods, and the Tool protocol's run()).
from schemas.tool_call import ToolCall


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


class FakeToolRouter:
    """Same shape as the real ToolRouter (a `.tools` dict keyed by tool
    name), but built from FakeTool instances instead of the real,
    GPU/filesystem-touching tool classes."""

    def __init__(self):

        self.tools = {
            "default": FakeTool("default"),
            "rag": FakeTool("rag", relay_verbatim=True),
            "note": FakeTool("note"),
        }
