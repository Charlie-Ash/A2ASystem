# Ochestrator class's main logic
# 1. Route to correct tools according to the prompt

from contextlib import AsyncExitStack
from pathlib import Path
from typing import Optional

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from orchestrator import config, memory_manager
from orchestrator.agents.tool_router import ToolRouter
from orchestrator.pipeline.llm import OrchestratorLLM
from orchestrator.pipeline.graph import build_graph
from schemas.tool_schema import build_tool_schema

class Orchestrator():

    def __init__(self, tool_router: ToolRouter, llm: OrchestratorLLM, graph, thread_id: str, exit_stack: AsyncExitStack):

        self.tool_router = tool_router
        self.llm = llm
        self.graph = graph

        # One Orchestrator instance is one continuous session (main.py's CLI
        # loop runs until "bye"), keyed against the graph's checkpointed
        # state -- that's what lets state["messages"] accumulate turn over
        # turn instead of resetting each call. Persisted to disk (see
        # memory_manager.load_or_create_thread_id) rather than a fresh
        # uuid4() every time, so a crashed/restarted process resumes the
        # same thread instead of opening a new, empty one.
        self.thread_id = thread_id

        # Holds the AsyncSqliteSaver's open connection (see create() below)
        # so it can be closed explicitly via aclose() once the session ends,
        # without forcing the whole main.py REPL loop inside one `async
        # with` block.
        self._exit_stack = exit_stack

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
        thread_id = memory_manager.load_or_create_thread_id()

        # AsyncSqliteSaver persists checkpointed graph state to disk (unlike
        # build_graph()'s default in-RAM MemorySaver), so conversation state
        # survives this process restarting -- entered via an AsyncExitStack,
        # not a plain `async with`, since its connection needs to stay open
        # for the whole session rather than one block here (closed later via
        # aclose(), called from main.py's "bye" flow).
        config.CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        exit_stack = AsyncExitStack()
        checkpointer = await exit_stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(str(config.CHECKPOINT_DB_PATH))
        )

        # The turn sequence (decide tool -> run tool -> generate reply ->
        # update memory) now lives in graph.py as a LangGraph StateGraph
        # instead of being written out inline here.
        graph = build_graph(llm, tool_router, checkpointer=checkpointer)

        return cls(tool_router, llm, graph, thread_id, exit_stack)

    # Runs one full turn through the graph built above and returns the final
    # reply text. Everything this used to do step-by-step -- tool decision,
    # tool execution, response generation, memory update -- now happens
    # inside the graph's nodes (see graph.py); this method just supplies the
    # user's message as the graph's starting state and reads the result back
    # out once the graph reaches END. Now async: the remote-agent tool nodes
    # do real network I/O, so the graph must be run via ainvoke() rather than
    # invoke() (see graph.py's run_remote_tool).
    async def run_orchestrator(self, user_message):

        # Named invoke_config (not config) so it doesn't shadow the
        # module-level `orchestrator.config` import above within this
        # function's scope.
        invoke_config = {"configurable": {"thread_id": self.thread_id}}
        result = await self.graph.ainvoke({"user_message": user_message}, config=invoke_config)
        return result["final_response"]

    # Called from main.py's "bye" flow once the user has answered the save prompt.
    def end_session(self, save: bool) -> Optional[Path]:

        if save:
            slug = self.llm.decide_chat_memory_filename()
            result = memory_manager.finalize_and_save(keep=True, new_name=slug)
        else:
            result = memory_manager.finalize_and_save(keep=False)

        # A clean exit -- the next process boot should start a genuinely new
        # session/thread rather than resuming this one (see
        # memory_manager.clear_thread_id).
        memory_manager.clear_thread_id()
        return result

    # Closes the persistent checkpointer's underlying sqlite connection.
    # Called from main.py right after end_session(), once the REPL loop is
    # about to exit -- not wrapped in try/finally around the whole loop, so
    # a crash simply leaves the connection to be closed by the OS on process
    # death, same "clean exit only" cleanup philosophy as chat_log.md/
    # thread_id.txt above.
    async def aclose(self) -> None:

        await self._exit_stack.aclose()
