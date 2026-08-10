# State schema for the Actions tool's own SUBGRAPH (see graph.py in this
# package). tool_call/tool_result are the same keys OrchestratorState already
# carries -- sharing their names is what lets this subgraph be registered
# directly as a node in the parent graph with no translation/wrapper node
# (see orchestrator/graph.py). Unlike RAGSubgraphState, "messages" is also
# shared: this agent's own content-generation call needs recent chat history
# to resolve references like "note down your previous answer to this
# question", which today's decide_tool (before this split) already had
# access to. request/generated_content/generated_file_name are private
# working state: LangGraph never merges them back into the parent, since the
# parent's schema doesn't declare them.
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from schemas.tool_call import ToolCall
from tools.base import ToolResult


class ActionsSubgraphState(TypedDict):

    # Shared with OrchestratorState: set by the parent's decide_tool node.
    tool_call: ToolCall

    # Shared with OrchestratorState: chat history before this turn, read by
    # generate_content to resolve references to earlier turns.
    messages: Annotated[list[AnyMessage], add_messages]

    # Private: the raw user request pulled out of tool_call.args by extract_request.
    request: str

    # Private: this agent's own LLM's generated note content/file_name.
    generated_content: str
    generated_file_name: str

    # Shared with OrchestratorState: read by the parent's generate_response node.
    tool_result: ToolResult
