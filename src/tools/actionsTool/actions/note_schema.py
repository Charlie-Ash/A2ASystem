# Structured-output schema for the Actions agent's own content-generation
# call (see graph.py's generate_content). Guided decoding (vLLM's
# structured_outputs) forces schema-conformant JSON regardless of model
# size, the same mechanism schemas/tool_schema.py's TOOL_SCHEMA already uses
# for the orchestrator's routing call -- so this stays reliable even on a
# small (1B) model.
from pydantic import BaseModel


class NoteContent(BaseModel):

    content: str
    file_name: str


NOTE_SCHEMA = NoteContent.model_json_schema()
