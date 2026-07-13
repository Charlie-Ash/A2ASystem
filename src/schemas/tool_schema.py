# Used as the dictionary that vLLM uses for guided decoding
from schemas.tool_call import ToolCall

# Derived directly from ToolCall so this schema can never drift out of sync with the Pydantic model.
TOOL_SCHEMA = ToolCall.model_json_schema()
