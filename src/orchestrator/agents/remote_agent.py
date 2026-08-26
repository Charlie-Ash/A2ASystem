# The orchestrator's actual A2A *client* plumbing: discovering a remote
# agent's AgentCard at startup, and sending it a turn's worth of user text
# over the real a2a-sdk client stack. This is what replaces
# ToolRouter's old in-process RAGTool()/ActionsTool() subgraph calls.
import logging
from dataclasses import dataclass

import httpx
from a2a.client import A2ACardResolver, Client, ClientConfig, ClientFactory
from a2a.client.helpers import create_text_message_object
from a2a.types import AgentCard, Task, TaskState, TextPart

from chat_history import history_to_chat_messages
from orchestrator.config import REMOTE_AGENT_TIMEOUT_SECONDS, RemoteAgentConfig
from tools.base import ToolResult

logger = logging.getLogger(__name__)


@dataclass
class RemoteAgent:
    """A remote agent this orchestrator successfully discovered at startup:
    its resolved AgentCard (used to build the tool-decision prompt's
    description of this agent -- see tool_router.py), a live a2a-sdk Client
    already wired to talk to it, and this agent's relay_verbatim policy."""

    name: str
    base_url: str
    agent_card: AgentCard
    client: Client
    relay_verbatim_on_success: bool


# Fetches this agent's AgentCard and builds a real a2a-sdk Client for it.
# Returns None (rather than raising) if the agent is unreachable -- the
# caller (ToolRouter.discover) logs this and excludes the agent from routing
# instead of refusing to start the whole orchestrator, since not every agent
# needs to be up for every dev/test run on this project's single GPU.
#
# httpx_client is normally left unset (production talks over real sockets);
# tests/fake_dependencies.py passes one backed by httpx.ASGITransport
# instead, so this exact function can be exercised against a real, in-process
# A2AFastAPIApplication with no real network involved.
async def discover_remote_agent(
    cfg: RemoteAgentConfig, httpx_client: "httpx.AsyncClient | None" = None
) -> "RemoteAgent | None":

    try:
        if httpx_client is not None:
            resolver = A2ACardResolver(httpx_client, cfg.url)
            card = await resolver.get_agent_card()
            client = await ClientFactory.connect(card, client_config=ClientConfig(httpx_client=httpx_client))
        else:
            # Explicit, generous timeout on both clients below -- without
            # this, ClientFactory.connect(card) (bare) and httpx.AsyncClient()
            # (bare) each fall back to httpx's stock 5-second default, which
            # is far too short for an LLM-backed agent call and was silently
            # truncating real, in-progress calls into false timeout errors
            # (see REMOTE_AGENT_TIMEOUT_SECONDS in orchestrator/config.py).
            timeout = httpx.Timeout(REMOTE_AGENT_TIMEOUT_SECONDS)

            async with httpx.AsyncClient(timeout=timeout) as resolver_client:
                resolver = A2ACardResolver(resolver_client, cfg.url)
                card = await resolver.get_agent_card()

            # Passing the already-resolved card (instead of the bare url) to
            # connect() skips re-fetching it -- one network round trip total.
            # This client is long-lived (stored on RemoteAgent, reused for
            # every call_remote_agent() call below), so it isn't closed here.
            persistent_client = httpx.AsyncClient(timeout=timeout)
            client = await ClientFactory.connect(card, client_config=ClientConfig(httpx_client=persistent_client))

    except Exception as e:
        logger.warning(f"Remote agent '{cfg.name}' unreachable at {cfg.url}: {e}")
        return None

    return RemoteAgent(cfg.name, cfg.url, card, client, cfg.relay_verbatim_on_success)


# Joins every TextPart in a list of Parts into one string -- both a Task's
# artifacts and a Message's parts share this same list[Part] shape, so this
# is reused for both branches below.
def _extract_text(parts) -> str:

    return "\n".join(part.root.text for part in parts if isinstance(part.root, TextPart))


# Sends one turn's worth of plain text to a remote agent over A2A and turns
# whatever comes back into this project's own ToolResult shape (see
# tools/base.py), so the rest of orchestrator/pipeline/graph.py doesn't need to know
# the call crossed a network boundary at all. Both RAG's and Actions'
# servers advertise capabilities.streaming=False, so send_message() always
# yields exactly one item: either a (Task, None) pair or a bare Message.
#
# history_messages (turns before this one, same slice generate_response
# already uses) is attached to the outgoing Message's free-form `metadata`
# field when given -- explicit payload, not a shared Python object, so it
# survives crossing a real process boundary. create_text_message_object only
# accepts role/content, so metadata has to be set by mutating the returned
# Message afterward. Every remote agent gets this regardless of whether it
# actually reads it (RAGAgentExecutor doesn't, ActionsAgentExecutor does) --
# call_remote_agent doesn't know or care which agent it's talking to.
async def call_remote_agent(agent: RemoteAgent, text: str, history_messages: list | None = None) -> ToolResult:

    message = create_text_message_object(content=text)
    if history_messages:
        message.metadata = {"history": history_to_chat_messages(history_messages)}

    try:
        async for event in agent.client.send_message(message):

            if isinstance(event, tuple):
                task, _update = event

                if task.status.state == TaskState.failed:
                    error_text = (
                        _extract_text(task.status.message.parts)
                        if task.status.message else f"{agent.name} agent task failed."
                    )
                    return {"output": error_text, "relay_verbatim": False}

                artifact_texts = [
                    _extract_text(artifact.parts) for artifact in (task.artifacts or [])
                ]
                return {
                    "output": "\n".join(artifact_texts),
                    "relay_verbatim": agent.relay_verbatim_on_success,
                }

            # Non-Task response: a bare Message straight back from the agent.
            return {
                "output": _extract_text(event.parts),
                "relay_verbatim": agent.relay_verbatim_on_success,
            }

    except Exception as e:
        # A request-time failure (agent went down mid-session, timed out,
        # etc) becomes an error ToolResult, same shape every other tool-error
        # path in this codebase already uses -- not a raised exception that
        # would crash the whole turn.
        return {"output": f"{agent.name} agent request failed: {e}", "relay_verbatim": False}

    return {"output": f"{agent.name} agent returned no result.", "relay_verbatim": False}
