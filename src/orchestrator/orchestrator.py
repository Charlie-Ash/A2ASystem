# Ochestrator class's main logic
# 1. Route to correct tools according to the prompt

from pathlib import Path
from typing import Optional

from orchestrator.tool_router import ToolRouter
from orchestrator.llm import OrchestratorLLM
from orchestrator import memory_manager
from orchestrator.graph import build_graph

class Orchestrator():

    def __init__(self):

        self.tool_router = ToolRouter()
        self.llm = OrchestratorLLM()

        # Fresh per-session memory file, reset on every boot (see memory_manager).
        memory_manager.init_system_memory()

        # The turn sequence (decide tool -> run tool -> generate reply ->
        # update memory) now lives in graph.py as a LangGraph StateGraph
        # instead of being written out inline here.
        self.graph = build_graph(self.llm, self.tool_router)

    # Runs one full turn through the graph built above and returns the final
    # reply text. Everything this used to do step-by-step -- tool decision,
    # tool execution, response generation, memory update -- now happens
    # inside the graph's nodes (see graph.py); this method just supplies the
    # user's message as the graph's starting state and reads the result back
    # out once the graph reaches END.
    def run_orchestrator(self, user_message):

        result = self.graph.invoke({"user_message": user_message})
        return result["final_response"]

    # Called from main.py's "bye" flow once the user has answered the save prompt.
    def end_session(self, save: bool) -> Optional[Path]:

        if save:
            slug = self.llm.decide_chat_memory_filename()
            return memory_manager.finalize_and_save(keep=True, new_name=slug)

        return memory_manager.finalize_and_save(keep=False)