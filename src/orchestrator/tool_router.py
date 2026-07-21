# Class that routes to the correct tool
from tools.defaultTool.tool_class import DefaultTool
from tools.ragTool.tool_class import RAGTool
from tools.notesTool.tool_class import NotesTool
from schemas.tool_call import ToolCall

class ToolRouter():

    def __init__(self):

        # Dictionary of all avialable tools
        self.tools = {

            "default": DefaultTool(),
            "rag": RAGTool(),
            "note": NotesTool()

        }

    def execute_tool(self, tool_call: ToolCall):

        # tool_call.tool is already validated against the 3 registered tools above,
        # so no "unknown tool" fallback is needed here anymore
        tool = self.tools[tool_call.tool]

        # Run each tool with their respective "run()" operation
        # This function should return the output (string format) of each tool, back to the orchestrator
        return tool.run(tool_call.args)