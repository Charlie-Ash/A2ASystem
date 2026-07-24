# Orchestrator prompt building
from orchestrator.memory_manager import EMPTY_MEMORY_PLACEHOLDER


# Builds the "CONVERSATION MEMORY SO FAR" block shared by both phase-1 and
# phase-2 prompts; omitted entirely while there's no memory yet, so early
# turns aren't cluttered with a placeholder line.
def _build_memory_section(memory_context: str) -> str:

    if not memory_context or memory_context == EMPTY_MEMORY_PLACEHOLDER:
        return ""

    return f"""
        -----------------------
        CONVERSATION MEMORY SO FAR
        -----------------------
        {memory_context}
    """


def build_tool_decision_prompt(user_message, memory_context):

    # First stage prompt: Tool Route
    # JSON shape enforced by guided decoding (see schemas/tool_schema.py),
    # so this prompt only needs to cover tool semantics, not output formatting.
    memory_section = _build_memory_section(memory_context)

    SYSTEM_PROMPT = f"""
        You are an orchestrator to a vast agent system.
        It is your role to decide on a suitble tool within the agent system to use in the user's work.
        Route tools that are connected to other agents accordingly from the user's message.
        {memory_section}
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

def build_response_prompt(user_message, tool_call, tool_result, memory_context):

    # Second-stage prompt: Plain-text reply
    memory_section = _build_memory_section(memory_context)

    SYSTEM_PROMPT = f"""
        You are the same orchestrator agent, now replying to the user directly.
        You already routed the user's message to the "{tool_call.tool}" tool and it has produced a result.
        {memory_section}
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


# Extracts the RAG tool's raw answer text out of RAGTool.run()'s formatted
# result string (a fixed marker sentence, blank line, then the answer).
def _extract_rag_answer(tool_result: str) -> str:

    parts = tool_result.split("\n\n", 1)
    return parts[1].strip() if len(parts) == 2 else tool_result.strip()


# Third-stage prompt: decides what's worth remembering from this turn.
def build_mem_update_prompt(user_message, tool_call, tool_result, final_response):

    result_text = str(tool_result) if tool_result is not None else "(no output)"

    if tool_call.tool == "rag":
        # RAG's raw answer is the ground truth for this turn -- prefer it over
        # final_response, which may have paraphrased or trimmed it.
        rag_answer = _extract_rag_answer(result_text)
        result_section = f"""
        This turn used the RAG tool. Structure your memory note as two labeled lines:
        Query: <the user's question, restated concisely>
        Answer: <the key facts from the RAG answer below, condensed>

        Raw RAG answer (verbatim, use as source of truth, not the final reply):
        {rag_answer}
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
