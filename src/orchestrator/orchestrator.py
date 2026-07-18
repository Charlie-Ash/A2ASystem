# Ochestrator class's main logic
# 1. Route to correct tools according to the prompt

from orchestrator.tool_router import ToolRouter
from orchestrator.llm import OrchestratorLLM

class Orchestrator():

    def __init__(self):

        self.tool_router = ToolRouter()
        self.llm = OrchestratorLLM()

    def run_orchestrator(self, user_message):

        # Ask LLM what tool should be used. A validated ToolCall is returned
        tool_call = self.llm.tool_decision(user_message)

        print("LLM tool decision: ", tool_call.tool)
        print("LLM tool argument: ", tool_call.args)

        # Execute tool
        result = self.tool_router.execute_tool(tool_call)

        # Ask LLM to turn the tool result into a natural-language reply
        final_response = self.llm.generate_response(user_message, tool_call, result)
        return final_response