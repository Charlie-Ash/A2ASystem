# State schema for the RAG tool's own SUBGRAPH (see graph.py in this package).
# tool_call/tool_result are the same keys OrchestratorState already carries --
# sharing their names is what lets this subgraph be registered directly as a
# node in the parent graph with no translation/wrapper node (see
# orchestrator/graph.py). query/retrieved_chunks are private working state:
# LangGraph never merges them back into the parent, since the parent's schema
# doesn't declare them.
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
