# Class that routes to the correct tool
from tools.defaultTool.tool_class import DefaultTool
from tools.ragTool.tool_class import RAGTool
from tools.notesTool.tool_class import NotesTool

class ToolRouter():

    def __init__(self):

        # Dictionary of all avialable tools
        self.tools = {

            "default": DefaultTool(),
            "rag": RAGTool(),
            "note": NotesTool()

        }