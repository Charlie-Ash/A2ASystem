# Shared helper for splicing graph-state chat history (OrchestratorState /
# ActionsSubgraphState's "messages" field, see orchestrator/pipeline/state.py and
# tools/actionsTool/actions/state.py) into a chat-template prompt as real
# prior turns, rather than a paraphrased summary. Used by both the
# orchestrator's own prompts (orchestrator/pipeline/prompts.py) and the Actions
# agent's content-generation prompt (tools/actionsTool/actions/graph.py), so
# both read the same conversation history the same way.
#
# The same {"role", "content"} dict shape this module already produces also
# doubles as the wire format for carrying history across an A2A call (see
# orchestrator/agents/remote_agent.py) -- it's already plain-JSON-safe and
# already capped, so no separate serialization format was needed.
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, convert_to_openai_messages

# Caps how many past turns get replayed into a prompt, so a long session's
# history can't blow a model's context budget.
MAX_HISTORY_TURNS = 8


def history_to_chat_messages(history_messages):

    recent = history_messages[-(MAX_HISTORY_TURNS * 2):]
    return convert_to_openai_messages(recent)


# The reverse of history_to_chat_messages: turns wire-format {"role",
# "content"} dicts back into LangChain messages. Only "user"/"assistant"
# ever appear on the wire (the source is always OrchestratorState.messages,
# which only ever holds Human/AI messages), so any other role is dropped
# rather than raised -- permissive, matching this module's existing style.
def chat_messages_to_history(wire_messages: list[dict]) -> list[AnyMessage]:

    role_to_cls = {"user": HumanMessage, "assistant": AIMessage}
    return [role_to_cls[m["role"]](content=m["content"]) for m in wire_messages if m.get("role") in role_to_cls]
