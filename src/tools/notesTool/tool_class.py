# Stores information into .txt file
from tools.notesTool.noteWriter import save_note


class NotesTool():

    def __init__(self):
        print("Notes tool initialized.")

    # Every tool's run() returns a ToolResult (see tools/base.py), passed to
    # the orchestrator to use to provide an answer. Note-saving is a status
    # action, not a user-facing answer, so relay_verbatim is always False here
    # -- the response-generation step composes a normal reply around it.
    def run(self, tool_args):

        content = tool_args.get("content", "")
        if not content:
            return {"output": "Notes tool error: no content was provided.", "relay_verbatim": False}

        # file_name is LLM-authored and optional; save_note() sanitizes it and
        # falls back to a timestamp-based name if it's missing or unusable.
        file_name = tool_args.get("file_name", "")

        try:
            saved_path = save_note(content, file_name)
        except Exception as e:
            return {"output": f"Notes tool error: failed to save note ({e})", "relay_verbatim": False}

        return {"output": f"Note saved to {saved_path}.", "relay_verbatim": False}
