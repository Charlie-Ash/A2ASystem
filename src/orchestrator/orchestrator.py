# Ochestrator class's main logic
# 1. Route to correct tools according to the prompt

import uuid
from pathlib import Path
from typing import Optional

from orchestrator.agents.tool_router import ToolRouter
from orchestrator.pipeline.llm import OrchestratorLLM
from orchestrator import memory_manager
from orchestrator.pipeline.graph import build_graph
from schemas.tool_schema import build_tool_schema

class Orchestrator():

    def __init__(self, tool_router: ToolRouter, llm: OrchestratorLLM, graph):

        self.tool_router = tool_router
        self.llm = llm
        self.graph = graph

        # One Orchestrator instance is one continuous session (main.py's CLI
        # loop runs until "bye"), so a single thread_id generated here is
        # enough to key every invoke() below against the same checkpointed
        # state -- that's what lets state["messages"] accumulate turn over
        # turn instead of resetting each call.
        self.thread_id = str(uuid.uuid4())

    # Async factory replacing what used to be all of __init__: ToolRouter.discover()
    # has to make real network calls (fetching each remote agent's AgentCard),
    # so building an Orchestrator is no longer a synchronous operation. This
    # is the only place ToolRouter/OrchestratorLLM/the graph get constructed --
    # __init__ above just wires already-built pieces together.
    @classmethod
    async def create(cls) -> "Orchestrator":

        tool_router = ToolRouter()
        await tool_router.discover()

        # The set of valid tool names (and therefore the guided-decoding
        # schema handed to OrchestratorLLM) isn't known until discovery above
        # has run -- see schemas/tool_schema.build_tool_schema.
        tool_schema = build_tool_schema(tool_router.all_tool_names())
        llm = OrchestratorLLM(tool_schema)

        # Fresh per-session chat log, reset on every boot (see memory_manager).
        memory_manager.init_chat_log()

        # The turn sequence (decide tool -> run tool -> generate reply ->
        # update memory) now lives in graph.py as a LangGraph StateGraph
        # instead of being written out inline here.
        graph = build_graph(llm, tool_router)

        return cls(tool_router, llm, graph)

    # Runs one full turn through the graph built above and returns the final
    # reply text. Everything this used to do step-by-step -- tool decision,
    # tool execution, response generation, memory update -- now happens
    # inside the graph's nodes (see graph.py); this method just supplies the
    # user's message as the graph's starting state and reads the result back
    # out once the graph reaches END. Now async: the remote-agent tool nodes
    # do real network I/O, so the graph must be run via ainvoke() rather than
    # invoke() (see graph.py's run_remote_tool).
    async def run_orchestrator(self, user_message):

        config = {"configurable": {"thread_id": self.thread_id}}
        result = await self.graph.ainvoke({"user_message": user_message}, config=config)
        return result["final_response"]

    # Called from main.py's "bye" flow once the user has answered the save prompt.
    def end_session(self, save: bool) -> Optional[Path]:

        if save:
            slug = self.llm.decide_chat_memory_filename()
            return memory_manager.finalize_and_save(keep=True, new_name=slug)

        return memory_manager.finalize_and_save(keep=False)
