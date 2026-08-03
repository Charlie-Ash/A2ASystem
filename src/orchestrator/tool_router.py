# Class that routes to the correct tool
from tools.defaultTool.tool_class import DefaultTool
from tools.ragTool.tool_class import RAGTool
from tools.notesTool.tool_class import NotesTool

class ToolRouter():

    def __init__(self):

        # Dictionary of all avialable tools that satisfy tools/base.py's Tool
        # protocol (a plain run(tool_args) -> ToolResult call).
        self.tools = {

            "default": DefaultTool(),
            "note": NotesTool()

        }

        # RAGTool no longer satisfies Tool -- its per-turn logic is a compiled
        # LangGraph subgraph (see tools/ragTool/graph.py), registered directly
        # as a node in the orchestrator's graph instead of being wrapped in a
        # run() call (see orchestrator/graph.py).
        self.rag_tool = RAGTool()
        self.rag_subgraph = self.rag_tool.subgraph