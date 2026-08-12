# Exercises orchestrator/agents/remote_agent.py's actual A2A client plumbing --
# discover_remote_agent (card fetch + client construction) and
# call_remote_agent (send a message, unpack the result into a ToolResult) --
# against a real, in-process A2AFastAPIApplication reached over
# httpx.ASGITransport instead of real sockets. No GPU, no real network, but
# genuinely the real a2a-sdk client and server code on both ends -- only the
# LangGraph subgraph underneath is a fake (same "real framework coverage, not
# more hand-fakes" approach as tests/test_rag_agent_executor.py.
import httpx
import pytest

from orchestrator.config import RemoteAgentConfig
from orchestrator.agents.remote_agent import call_remote_agent, discover_remote_agent

from fake_dependencies import (
    build_fake_failing_subgraph,
    build_fake_rag_subgraph,
    build_fake_remote_agent,
)
from tools.ragTool.a2a.a2a_server import build_agent_card as build_rag_agent_card
from tools.ragTool.a2a.agent_executor import RAGAgentExecutor
from tools.ragTool.rag.state import RAGSubgraphState


async def test_discover_remote_agent_succeeds_against_a_real_in_process_server():

    agent = await build_fake_remote_agent(
        "rag", RAGAgentExecutor(build_fake_rag_subgraph()), build_rag_agent_card, relay_verbatim_on_success=True
    )

    assert agent.name == "rag"
    assert agent.agent_card.skills[0].id == "answer_from_company_documents"


async def test_discover_remote_agent_returns_none_for_an_unreachable_agent():

    # Simulates "agent process isn't running" without touching real
    # network/DNS: MockTransport raises a connection error for every
    # request, same failure shape httpx would raise against a dead port.
    def _always_fails(request):
        raise httpx.ConnectError("connection refused (simulated)")

    httpx_client = httpx.AsyncClient(transport=httpx.MockTransport(_always_fails))
    cfg = RemoteAgentConfig(name="actions", url="http://actions.test", relay_verbatim_on_success=False)

    agent = await discover_remote_agent(cfg, httpx_client=httpx_client)

    # Graceful degradation, not a raised exception -- ToolRouter.discover()
    # relies on exactly this to skip an unreachable agent and keep starting.
    assert agent is None


async def test_call_remote_agent_round_trips_text_and_uses_the_agents_relay_verbatim_config():

    agent = await build_fake_remote_agent(
        "rag",
        RAGAgentExecutor(build_fake_rag_subgraph(canned_output="the answer")),
        build_rag_agent_card,
        relay_verbatim_on_success=True,
    )

    tool_result = await call_remote_agent(agent, "what is Pete's favorite subject?")

    assert tool_result == {"output": "the answer", "relay_verbatim": True}


async def test_call_remote_agent_forces_relay_verbatim_false_when_the_agent_task_fails():

    agent = await build_fake_remote_agent(
        "rag",
        RAGAgentExecutor(build_fake_failing_subgraph(RAGSubgraphState, error_message="boom")),
        build_rag_agent_card,
        relay_verbatim_on_success=True,  # even so, a failed task must still force this to False
    )

    tool_result = await call_remote_agent(agent, "anything")

    assert tool_result["relay_verbatim"] is False
    assert "boom" in tool_result["output"]
