# Owns every tool the orchestrator can route to: the local, in-process
# "default" tool (kept in-process on purpose -- no reason to network a stub,
# see PROJECT_SPRINT_NOTES.md), and whatever remote A2A agents got
# discovered/configured at startup (see orchestrator/config.py). This
# replaces the old ToolRouter, which constructed RAGTool()/ActionsTool()
# directly and ran their subgraphs in-process.
from tools.defaultTool.tool_class import DefaultTool

from orchestrator import config
from orchestrator.agents.remote_agent import RemoteAgent, discover_remote_agent


class ToolRouter():

    def __init__(self):

        # Dictionary of all available in-process tools that satisfy
        # tools/base.py's Tool protocol (a plain run(tool_args) -> ToolResult
        # call). Only "default" belongs here now -- RAG/Actions are real
        # network agents, held in self.remote_agents instead.
        self.tools = {

            "default": DefaultTool()

        }

        # Populated by discover() below. Keyed by the routing name each
        # RemoteAgentConfig entry was given (e.g. "rag", "actions"), not
        # hardcoded attributes -- however many agents orchestrator/config.py
        # lists, that's how many entries show up here.
        self.remote_agents: dict[str, RemoteAgent] = {}

    # Resolves every agent listed in config.REMOTE_AGENTS by actually
    # fetching its AgentCard over the network. An unreachable agent is
    # logged and skipped (not fatal) -- see remote_agent.discover_remote_agent
    # for why. Must be awaited once before the graph is built (orchestrator.py's
    # Orchestrator.create() is the only caller).
    async def discover(self):

        for agent_config in config.REMOTE_AGENTS:

            agent = await discover_remote_agent(agent_config)

            if agent is not None:
                self.remote_agents[agent.name] = agent

    # Every valid ToolCall.tool value right now: the local tool(s) plus
    # whichever remote agents were actually reachable at startup. Used to
    # build the guided-decoding schema (schemas/tool_schema.build_tool_schema)
    # so the LLM can never pick a tool name that isn't really available.
    def all_tool_names(self) -> list[str]:

        return list(self.tools) + list(self.remote_agents)

    # Builds the "what can each tool do" descriptions the tool-decision
    # prompt shows the LLM (see orchestrator/pipeline/prompts.py). "default" gets a
    # short hand-written description since it has no AgentCard; every remote
    # agent's description/example comes straight from its own AgentCard's
    # first skill -- discovered from the network, not hand-typed here, so it
    # can't drift out of sync with what that agent actually does.
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
