# Orchestrator prompt building
from chat_history import history_to_chat_messages as _history_to_chat_messages


def build_tool_decision_prompt(user_message, history_messages, tool_descriptions):

    # First stage prompt: Tool Route
    # JSON shape enforced by guided decoding (see schemas/tool_schema.py),
    # so this prompt only needs to cover tool semantics, not output formatting.
    # tool_descriptions comes from ToolRouter.describe_tools_for_prompt() --
    # a list of {"name", "description", "example_user_message"} dicts, one
    # per available tool. "default"'s is hand-written (no AgentCard to pull
    # from); every remote agent's comes straight from its own discovered
    # AgentCard, so this prompt can't drift out of sync with what that agent
    # actually advertises. args is always {} in every example now -- no tool
    # reads tool_call.args anymore (remote agents just get the raw user
    # message as A2A message text; "default" ignores its args entirely).
    tools_block = "\n\n".join(
        f"{i}. {tool['name']}\n{tool['description']}\nargs: {{}}"
        for i, tool in enumerate(tool_descriptions, start=1)
    )

    examples_block = "\n\n".join(
        f"""User: {tool['example_user_message']}
        Output:
        {{
            "tool": "{tool['name']}",
            "action": "run",
            "args": {{}}
        }}"""
        for tool in tool_descriptions
    )

    SYSTEM_PROMPT = f"""
        You are an orchestrator to a vast agent system.
        It is your role to decide on a suitble tool within the agent system to use in the user's work.
        Route tools that are connected to other agents accordingly from the user's message.
        -----------------------
        TOOLS AVAILABLE
        -----------------------

        {tools_block}

        -----------------------
        EXAMPLES
        -----------------------

        {examples_block}
    """

    # Have the system prompt and user message split, avoiding using a single, excessivly long prompt that may cause unexpected behaviors.
    # Prior turns (if any) are spliced in between as real chat turns, so the
    # model sees actual history instead of a summary folded into the system prompt.
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        *_history_to_chat_messages(history_messages),
        {
            "role": "user",
            "content": user_message,
        }
    ]

    return messages

def build_response_prompt(user_message, history_messages, tool_call, tool_result):

    # Second-stage prompt: Plain-text reply.
    # tool_result is a ToolResult (see tools/base.py): {"output": str, "relay_verbatim": bool}.
    if tool_result["relay_verbatim"]:
        # The tool already produced a complete, user-ready answer (e.g. RAG) --
        # tell the model to show it as-is instead of paraphrasing it away.
        instruction = (
            "The tool result below was already fully generated as a complete answer. "
            "Relay it to the user as-is without rewriting or summarizing it, then ask "
            "what they'd like help with next."
        )
    else:
        instruction = "Using the tool result below, write a concise, helpful reply to the user."

    SYSTEM_PROMPT = f"""
        You are the same orchestrator agent, now replying to the user directly.
        You already routed the user's message to the "{tool_call.tool}" tool and it has produced a result.
        {instruction}
        Do not mention tool names, JSON, or internal routing details.

        Tool result:
        {tool_result["output"]}
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        *_history_to_chat_messages(history_messages),
        {
            "role": "user",
            "content": user_message,
        }
    ]

    return messages


# Third-stage prompt: decides what's worth remembering from this turn.
# tool_result is a ToolResult (see tools/base.py): {"output": str, "relay_verbatim": bool}.
def build_mem_update_prompt(user_message, tool_call, tool_result, final_response):

    result_text = tool_result["output"]

    # Known soft spot: this string comparison assumes the RAG agent is
    # always configured under the name "rag" (orchestrator/config.py's
    # REMOTE_AGENTS default). If an operator renames it there, this branch
    # silently stops firing -- not fixed here, since the tool name is no
    # longer a closed enum this file could validate against.
    if tool_call.tool == "rag":
        # RAG's raw answer is the ground truth for this turn -- prefer it over
        # final_response, which may have paraphrased or trimmed it. No more
        # string-splitting needed: tool_result["output"] is already just the
        # answer text, since relay_verbatim is carried as its own field now.
        result_section = f"""
        This turn used the RAG tool. Structure your memory note as two labeled lines:
        Query: <the user's question, restated concisely>
        Answer: <the key facts from the RAG answer below, condensed>

        Raw RAG answer (verbatim, use as source of truth, not the final reply):
        {result_text}
        """
    else:
        result_section = f"""
        Tool used: {tool_call.tool}
        Tool result: {result_text}
        """

    SYSTEM_PROMPT = f"""
        You are the memory-keeper for an ongoing conversation with an agent system.
        Write a concise memory note (2-4 sentences, or the Query/Answer format if
        instructed below) summarizing what was asked and what was learned or decided
        this turn. This note will be read back by yourself in later turns, so prioritize
        concrete facts, names, decisions, and any retrieved answers over vague description.
        Do not restate these instructions. Output only the note text, no headers.

        {result_section}

        Orchestrator's final reply to the user this turn: {final_response}
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


# Fourth-stage prompt: names the saved memory file at end-of-session ("bye" + "y").
def build_filename_prompt(memory_content):

    SYSTEM_PROMPT = f"""
        You are naming a saved chat memory file for later reference.
        Based on the conversation memory below, output a short topic slug: 2-5 words,
        lowercase, words separated by underscores, no punctuation, no file extension,
        and do not include the words "chat" or "points" (a suffix is appended automatically).
        Output ONLY the slug, nothing else.

        Conversation memory:
        {memory_content}
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": "Provide the slug now.",
        }
    ]

    return messages
