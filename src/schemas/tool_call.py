# Single source of truth for what a "tool call" looks like
from pydantic import BaseModel
from typing import Literal

class ToolCall(BaseModel):

    # Must match one of the 3 tools registered in ToolRouter
    tool: Literal[
        "default",
        "rag",
        "note"
    ]

    action: Literal["run"]

    args: dict
