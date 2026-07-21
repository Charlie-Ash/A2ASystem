# Uesed for testing
class DefaultTool():
    
    def __init__(self):

        print("Default tool initialized.")

    # The run() function of each tool returns a string, passed to the orchestrator to use to provide and answer.
    def run(self, tool_args):

        print("Default tool running...")
        return "Default tool has completed"