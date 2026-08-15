# State schema for the Actions tool's own SUBGRAPH (see graph.py in this
# package). tool_call/tool_result/messages happen to share their names with
# OrchestratorState's fields (see orchestrator/pipeline/state.py) -- originally
# because this subgraph used to be registered directly as a node inside the
# orchestrator's own graph, sharing its checkpointed chat history so
# generate_content could resolve references like "note down your previous
# answer to this question". Since the A2A rework, this subgraph is only ever
# invoked by this package's own standalone A2A server (a2a/agent_executor.py),
# which reconstructs "messages" itself from history explicitly attached to
# the incoming A2A message (see that file's module comment) rather than
# inheriting a shared checkpointer -- same end result, different plumbing.
# request/generated_content/generated_file_name are private working state
# either way: LangGraph never merges undeclared fields back to a caller.
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
