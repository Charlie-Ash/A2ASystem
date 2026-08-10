# Class that routes to the correct tool
from tools.defaultTool.tool_class import DefaultTool
from tools.ragTool.rag.tool_class import RAGTool
from tools.actionsTool.actions.tool_class import ActionsTool

class ToolRouter():

    def __init__(self):

        # Dictionary of all avialable tools that satisfy tools/base.py's Tool
        # protocol (a plain run(tool_args) -> ToolResult call).
        self.tools = {

            "default": DefaultTool()

        }

        # RAGTool/ActionsTool don't satisfy Tool -- their per-turn logic is a
        # compiled LangGraph subgraph each (see tools/ragTool/rag/graph.py,
        # tools/actionsTool/actions/graph.py), registered directly as nodes
        # in the orchestrator's graph instead of being wrapped in a run()
        # call (see orchestrator/graph.py).
        self.rag_tool = RAGTool()
        self.rag_subgraph = self.rag_tool.subgraph

        self.actions_tool = ActionsTool()
        self.actions_subgraph = self.actions_tool.subgraph