# Shared helper for splicing graph-state chat history (OrchestratorState /
# ActionsSubgraphState's "messages" field, see orchestrator/state.py and
# tools/actionsTool/actions/state.py) into a chat-template prompt as real
# prior turns, rather than a paraphrased summary. Used by both the
# orchestrator's own prompts (orchestrator/prompts.py) and the Actions
# agent's content-generation prompt (tools/actionsTool/actions/graph.py), so
# both read the same conversation history the same way.
from langchain_core.messages import convert_to_openai_messages

# Caps how many past turns get replayed into a prompt, so a long session's
# history can't blow a model's context budget.
MAX_HISTORY_TURNS = 8


def history_to_chat_messages(history_messages):

    recent = history_messages[-(MAX_HISTORY_TURNS * 2):]
    return convert_to_openai_messages(recent)
