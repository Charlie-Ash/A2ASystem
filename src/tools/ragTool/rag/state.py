# State schema for the RAG tool's own SUBGRAPH (see graph.py in this package).
# tool_call/tool_result happen to share their names with OrchestratorState's
# fields (see orchestrator/pipeline/state.py) -- a leftover from when this
# subgraph used to be registered directly as a node inside the orchestrator's
# own graph. Since the A2A rework, this subgraph is only ever invoked by this
# package's own standalone A2A server (a2a/agent_executor.py), as a top-level
# graph in its own right -- there's no longer a literal parent graph to share
# keys with. query/retrieved_chunks are private working state either way:
# LangGraph never merges undeclared fields back to a caller.
from typing import TypedDict

from schemas.tool_call import ToolCall
from tools.base import ToolResult


class RAGSubgraphState(TypedDict):

    # Shared with OrchestratorState: set by the parent's decide_tool node.
    tool_call: ToolCall

    # Private: the query text pulled out of tool_call.args by extract_query.
    query: str

    # Private: chunks returned by retrieve.
    retrieved_chunks: list[str]

    # Shared with OrchestratorState: read by the parent's generate_response node.
    tool_result: ToolResult
