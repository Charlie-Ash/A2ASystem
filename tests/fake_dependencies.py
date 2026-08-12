# Stand-ins for OrchestratorLLM and the local tool, used by the graph tests
# so they can run without a GPU. The real RAGTool/ActionsTool each load a
# whole vLLM engine (RAGTool a vector index too) in their __init__, so they
# must never be constructed here -- these fakes only need to satisfy the same
# shapes graph.py calls against (OrchestratorLLM's phase methods, and the
# Tool protocol's run()).
#
# Remote agents (RAG/Actions) are no longer faked with a stand-in subgraph
# wired straight into the graph -- they're real A2A network calls now (see
# orchestrator/agents/remote_agent.py). So instead of faking the *call*, this file
# fakes the *server on the other end*: a real, in-process
# A2AFastAPIApplication wrapping the existing fake subgraphs, reached over
# httpx.ASGITransport instead of real sockets. graph.py's remote-tool nodes
# then run through the exact same discover_remote_agent()/call_remote_agent()
# code paths production uses -- no separate fake client-call logic to keep in
# sync with the real one.
import httpx
from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from langgraph.graph import StateGraph, START, END

from orchestrator.config import RemoteAgentConfig
from orchestrator.agents.remote_agent import RemoteAgent, discover_remote_agent
from schemas.tool_call import ToolCall
from tools.actionsTool.a2a.a2a_server import build_agent_card as build_actions_agent_card
from tools.actionsTool.a2a.agent_executor import ActionsAgentExecutor
from tools.actionsTool.actions.state import ActionsSubgraphState
from tools.ragTool.a2a.a2a_server import build_agent_card as build_rag_agent_card
from tools.ragTool.a2a.agent_executor import RAGAgentExecutor
from tools.ragTool.rag.state import RAGSubgraphState


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

    def tool_decision(self, user_message, history_messages, tool_descriptions) -> ToolCall:

        self.tool_decision_calls.append((user_message, history_messages, tool_descriptions))
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
# function), so the fake RAG server underneath exercises the real
# subgraph-as-node mechanics without ever touching vLLM/Qdrant/llama-index.
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
# real output, same reasoning as build_fake_rag_subgraph above.
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


# A subgraph whose single node always raises -- used to exercise the
# executors' own try/except-into-updater.failed() path (see
# tools/*/a2a/agent_executor.py), and from there, call_remote_agent's
# task-failed handling (see orchestrator/agents/remote_agent.py).
def build_fake_failing_subgraph(state_cls, error_message="boom"):

    def fake_failing_run(state) -> dict:
        raise RuntimeError(error_message)

    sub = StateGraph(state_cls)
    sub.add_node("fake_failing_run", fake_failing_run)
    sub.add_edge(START, "fake_failing_run")
    sub.add_edge("fake_failing_run", END)
    return sub.compile()


# Spins up a real A2AFastAPIApplication wrapping the given (real)
# AgentExecutor + fake subgraph, reached over httpx.ASGITransport, then
# resolves it into a RemoteAgent via the exact same discover_remote_agent()
# the orchestrator uses in production. This is what lets graph.py's
# make_remote_tool_node/call_remote_agent be tested against the real
# a2a-sdk client/server stack end-to-end, GPU-free.
async def build_fake_remote_agent(name, agent_executor, build_agent_card_fn, relay_verbatim_on_success):

    agent_card = build_agent_card_fn("fake-host", 0)
    handler = DefaultRequestHandler(agent_executor=agent_executor, task_store=InMemoryTaskStore())
    app = A2AFastAPIApplication(agent_card=agent_card, http_handler=handler).build()

    # base_url is never actually resolved over DNS/sockets -- ASGITransport
    # dispatches every request straight into `app` regardless of host/port,
    # so this only needs to be a syntactically valid URL.
    httpx_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=f"http://{name}.test")

    cfg = RemoteAgentConfig(name=name, url=f"http://{name}.test", relay_verbatim_on_success=relay_verbatim_on_success)
    agent = await discover_remote_agent(cfg, httpx_client=httpx_client)
    assert agent is not None, f"fake '{name}' agent discovery failed -- check the fake server wiring above"
    return agent


class FakeToolRouter:
    """Same shape as the real ToolRouter (a `.tools` dict keyed by tool
    name, plus a `.remote_agents` dict keyed by agent name), but built from
    FakeTool instances and fake-subgraph-backed A2A servers instead of the
    real, GPU/filesystem-touching tool classes.

    Construction is async (discover_remote_agent does real ASGI round trips)
    -- use `await FakeToolRouter.create()` rather than the plain constructor.
    """

    def __init__(self):

        self.tools = {
            "default": FakeTool("default"),
        }
        self.remote_agents: dict[str, RemoteAgent] = {}

        # Populated by create() below; kept here too under their old names
        # so existing tests reading tool_router.rag_received_args/
        # actions_received_args don't need to change shape.
        self.rag_received_args = {"args": None}
        self.actions_received_args = {"args": None}
        self.actions_received_messages = {"messages": None}

    @classmethod
    async def create(cls) -> "FakeToolRouter":

        self = cls()

        rag_subgraph = build_fake_rag_subgraph(
            canned_output="rag-output", relay_verbatim=True, received_args_sink=self.rag_received_args
        )
        rag_agent = await build_fake_remote_agent(
            "rag", RAGAgentExecutor(rag_subgraph), build_rag_agent_card, relay_verbatim_on_success=True
        )
        self.remote_agents["rag"] = rag_agent

        actions_subgraph = build_fake_actions_subgraph(
            canned_output="actions-output",
            relay_verbatim=False,
            received_args_sink=self.actions_received_args,
            received_messages_sink=self.actions_received_messages,
        )
        actions_agent = await build_fake_remote_agent(
            "actions", ActionsAgentExecutor(actions_subgraph), build_actions_agent_card,
            relay_verbatim_on_success=False,
        )
        self.remote_agents["actions"] = actions_agent

        return self

    def all_tool_names(self) -> list[str]:

        return list(self.tools) + list(self.remote_agents)

    def describe_tools_for_prompt(self) -> list[dict]:

        descriptions = [{
            "name": "default",
            "description": "Use for general testing or unclear intent.",
            "example_user_message": "hello",
        }]

        for agent in self.remote_agents.values():
            skill = agent.agent_card.skills[0]
            example = skill.examples[0] if skill.examples else agent.agent_card.description
            descriptions.append({
                "name": agent.name,
                "description": skill.description,
                "example_user_message": example,
            })

        return descriptions
