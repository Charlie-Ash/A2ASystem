# Single source of truth for what a "tool call" looks like
from pydantic import BaseModel
from typing import Literal

class ToolCall(BaseModel):

    # Which of the tools registered in ToolRouter to use. Not a Literal[...]
    # anymore -- once "default" is local but "rag"/"actions" (and whatever
    # else gets configured) are remote A2A agents, the valid set is only
    # known once ToolRouter has discovered them at startup, not at import
    # time. The guided-decoding schema (see schemas/tool_schema.py) still
    # constrains this to exactly the discovered names, so the LLM can't
    # actually produce a value outside that set -- just no longer a
    # hardcoded enum here.
    tool: str

    action: Literal["run"]

    # No longer read by any tool -- both DefaultTool.run() and the remote
    # agents (which just read the raw A2A message text server-side) ignore
    # this. Kept for schema shape-compatibility / future use.
    args: dict
