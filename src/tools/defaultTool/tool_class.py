# Uesed for testing
class DefaultTool():

    def __init__(self):

        print("Default tool initialized.")

    # Every tool's run() returns a ToolResult (see tools/base.py), passed to
    # the orchestrator to use to provide an answer.
    def run(self, tool_args):

        print("Default tool running...")
        return {"output": "Default tool has completed.", "relay_verbatim": False}