# Orchestrator prompt building

def build_tool_decision_prompt(user_message):

    # First stage prompt: Tool Route
    # JSON shape enforced by guided decoding (see schemas/tool_schema.py),
    # so this prompt only needs to cover tool semantics, not output formatting.
    SYSTEM_PROMPT = f"""
        You are an orchestrator to a vast agent system.
        It is your role to decide on a suitble tool within the agent system to use in the user's work.
        Route tools that are connected to other agents accordingly from the user's message.

        -----------------------
        TOOLS AVAILABLE
        -----------------------

        1. default
        Use for general testing or unclear intent.
        args: {{}}

        2. rag
        Use for document Q&A.
        args: {{
            "query": string
        }}

        3. note
        Use for saving information.
        args: {{
            "content": string,
            "file_name": string
        }}

        ("file_name" is a short, filesystem-safe title for the note (lowercase,
        words separated by underscores, no extension) -- the note tool appends
        "_notes.txt" itself.)

        -----------------------
        EXAMPLES
        -----------------------

        User: What is Pete's favorite subject?
        Output:
        {{
            "tool": "rag",
            "action": "run",
            "args": {{
                "query": "What is Pete's favorite subject?"
                }}
        }}

        User: Remember Pete likes astronomy
        Output:
        {{
            "tool": "note",
            "action": "run",
            "args": {{
                "content": "Pete likes astronomy",
                "file_name": "pete_astronomy"
                }}
        }}

        User: hello
        Output:
        {{
            "tool": "default",
            "action": "run",
            "args": {{}}
        }}
    """

    # Have the system prompt and user message split, avoiding using a single, excessivly long prompt that may cause unexpected behaviors
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_message,
        }
    ]

    return messages

def build_response_prompt(user_message, tool_call, tool_result):

    # Second-stage prompt: Plain-text reply
    SYSTEM_PROMPT = f"""
        You are the same orchestrator agent, now replying to the user directly.
        You already routed the user's message to the "{tool_call.tool}" tool and it has produced a result.

        Using the tool result below, write a concise, helpful reply to the user.
        Do not mention tool names, JSON, or internal routing details.

        Tool result:
        {tool_result}
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_message,
        }
    ]

    return messages
