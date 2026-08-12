# Used as the dictionary that vLLM uses for guided decoding
from typing import Sequence

from schemas.tool_call import ToolCall


# Builds the guided-decoding schema from whatever tool names ToolRouter
# actually discovered/configured at startup (see orchestrator/agents/tool_router.py),
# rather than a hardcoded enum. Still derived from ToolCall.model_json_schema()
# so "action"/"args" can never drift out of sync with the pydantic model --
# only the "tool" property's allowed values are injected here, since that's
# the one piece that isn't known until the registry resolves.
def build_tool_schema(tool_names: Sequence[str]) -> dict:

    schema = ToolCall.model_json_schema()
    schema["properties"]["tool"]["enum"] = list(tool_names)
    return schema
