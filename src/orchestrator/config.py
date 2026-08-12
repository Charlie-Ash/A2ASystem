# Orchestrator-side config: which remote A2A agents to discover at startup,
# and how each one's results should be relayed. Same os.environ.get(...)
# convention already used in tools/ragTool/config.py / tools/actionsTool/config.py.

import os
from dataclasses import dataclass


@dataclass
class RemoteAgentConfig:
    """One entry from REMOTE_AGENTS: a short routing name, the agent's base
    URL, and whether a *successful* result from this agent should be relayed
    to the user as-is (True) or composed into a reply by generate_response
    (False). A failed/errored call always forces relay_verbatim=False
    regardless of this default -- see orchestrator/agents/remote_agent.py."""

    name: str
    url: str
    relay_verbatim_on_success: bool


# "name@url,name@url,..." -- e.g. "rag@http://localhost:8001,actions@http://localhost:8002".
# Adding/removing an agent is a one-line env var edit, not a code change.
# Matches the ports the standalone servers already default to
# (tools/ragTool/config.py's RAG_A2A_PORT=8001, tools/actionsTool/config.py's
# ACTIONS_A2A_PORT=8002) -- "localhost" here on purpose, not "0.0.0.0": that's
# a bind address for the servers, not something a client should connect to.
_DEFAULT_REMOTE_AGENTS = "rag@http://localhost:8001,actions@http://localhost:8002"

# Comma-separated list of agent names (from the entries above) whose
# successful results should be relayed verbatim. Anyone not listed defaults
# to relay_verbatim_on_success=False (composed into a reply), the safer
# default for an agent whose output style isn't known in advance.
_DEFAULT_REMOTE_AGENTS_VERBATIM = "rag"


def _parse_remote_agents(raw: str, verbatim_raw: str) -> list[RemoteAgentConfig]:

    verbatim_names = {name.strip() for name in verbatim_raw.split(",") if name.strip()}

    agents = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, _, url = entry.partition("@")
        agents.append(RemoteAgentConfig(
            name=name.strip(),
            url=url.strip(),
            relay_verbatim_on_success=name.strip() in verbatim_names,
        ))

    return agents


REMOTE_AGENTS = _parse_remote_agents(
    os.environ.get("REMOTE_AGENTS", _DEFAULT_REMOTE_AGENTS),
    os.environ.get("REMOTE_AGENTS_VERBATIM", _DEFAULT_REMOTE_AGENTS_VERBATIM),
)
