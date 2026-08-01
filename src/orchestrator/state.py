# The shared "form" that gets passed between every node in the orchestrator's
# LangGraph graph (see graph.py). Each node reads whatever fields it needs and
# returns a partial dict of the fields it wants to update; LangGraph merges
# that dict into this state before running the next node. A TypedDict has no
# runtime behavior of its own (no validation, no defaults) -- it only exists
# to give editors/type-checkers something to check node code against.
from typing import TypedDict

from schemas.tool_call import ToolCall
from tools.base import ToolResult


class OrchestratorState(TypedDict):

    # The raw text the user typed this turn.
    user_message: str

    # Set by the "decide_tool" node: which tool to run and with what args.
    tool_call: ToolCall

    # Set by whichever tool node ran: the tool's output for this turn.
    tool_result: ToolResult

    # Set by the "generate_response" node: the reply shown to the user.
    final_response: str
