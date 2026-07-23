# Stores information into .txt file
from tools.notesTool.noteWriter import save_note


class NotesTool():

    def __init__(self):
        print("Notes tool initialized.")

    # The run() function of each tool returns a string, passed to the orchestrator to use to provide and answer.
    def run(self, tool_args):

        content = tool_args.get("content", "")
        if not content:
            return "Notes tool error: no content was provided."

        # file_name is LLM-authored and optional; save_note() sanitizes it and
        # falls back to a timestamp-based name if it's missing or unusable.
        file_name = tool_args.get("file_name", "")

        try:
            saved_path = save_note(content, file_name)
        except Exception as e:
            return f"Notes tool error: failed to save note ({e})"

        return f"[Notes tool completed] Note saved to {saved_path}."
