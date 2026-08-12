# Builds the Actions tool's own SUBGRAPH. Mirrors orchestrator/pipeline/graph.py's
# build_graph(llm, tool_router) and ragTool/rag/graph.py's
# build_rag_subgraph(): takes the already-loaded llm/sampling_params as
# parameters instead of constructing them itself, so this module's wiring
# can be tested without a GPU (see tests/fake_dependencies.py).
from typing import TYPE_CHECKING

from langgraph.graph import StateGraph, START, END

from chat_history import history_to_chat_messages
from tools.actionsTool.actions.note_schema import NoteContent
from tools.actionsTool.actions.noteWriter import save_note
from tools.actionsTool.actions.state import ActionsSubgraphState

# Imported only for "LLM"/"SamplingParams" to be resolved as types for
# string-quoted annotations on build_actions_subgraph.
if TYPE_CHECKING:
    from vllm import LLM, SamplingParams

SYS_PROMPT = (
    "You are a note-taking assistant. Given the user's request below, write "
    "the note content that should be saved, and choose a short, "
    "filesystem-safe file_name slug (lowercase, words separated by "
    "underscores, no extension, no spaces or punctuation). If the request "
    "refers to something said earlier in the conversation (e.g. 'note down "
    "your previous answer'), use the conversation history to resolve it."
)


def build_actions_subgraph(llm: "LLM", sampling_params: "SamplingParams"):
    """Assemble and compile the Actions tool's subgraph: extract_request ->
    generate_content -> save_note, with an empty-request short-circuit
    straight to END and a generation-failure short-circuit that skips
    save_note.

    Compiled with no explicit checkpointer, same reasoning as RAG's
    build_rag_subgraph(): inherits the parent orchestrator graph's
    MemorySaver when run as a node there. Unlike RAG, this subgraph declares
    "messages" as a shared state key (see state.py) specifically so it
    inherits that checkpointed history too, not just the checkpointer
    object -- generate_content needs it to resolve references to earlier
    turns.
    """

    graph = StateGraph(ActionsSubgraphState)

    # Node: pull the request out of this turn's tool_call.args. If it's
    # missing, also set the same error tool_result the rest of the graph
    # would otherwise have had to special-case downstream -- route_after_extract
    # below sends this straight to END instead of attempting generation.
    def extract_request(state: ActionsSubgraphState) -> dict:

        request = state["tool_call"].args.get("request", "")

        if not request:
            return {
                "request": request,
                "tool_result": {"output": "Actions tool error: no request was provided.", "relay_verbatim": False},
            }

        return {"request": request}

    def route_after_extract(state: ActionsSubgraphState) -> str:

        return "has_request" if state["request"] else "empty_request"

    # Node: this agent's own LLM writes the note's content and file_name
    # from the raw request, using recent chat history (turns before this
    # one -- state["messages"] would already include this turn's just-appended
    # HumanMessage, same convention orchestrator/pipeline/graph.py's
    # generate_response uses) so it can resolve references like "note down
    # your previous answer". In practice, via this package's standalone A2A
    # server, "messages" is never populated at all (see a2a/agent_executor.py's
    # "known limitation" comment) -- this history-aware behavior only fires
    # if something invokes this subgraph with real prior turns in state.
    def generate_content(state: ActionsSubgraphState) -> dict:

        history_messages = state.get("messages", [])[:-1]

        messages = [
            {"role": "system", "content": SYS_PROMPT},
            *history_to_chat_messages(history_messages),
            {"role": "user", "content": state["request"]},
        ]

        try:
            outputs = llm.chat(messages, sampling_params)
            note = NoteContent.model_validate_json(outputs[0].outputs[0].text)
        except Exception as e:
            return {"tool_result": {"output": f"Actions tool error: generation failed ({e})", "relay_verbatim": False}}

        return {"generated_content": note.content, "generated_file_name": note.file_name}

    # Routing function: generate_content sets tool_result itself on failure
    # instead of raising, so route on whether generation actually produced
    # content rather than on an exception.
    def route_after_generate(state: ActionsSubgraphState) -> str:

        return "success" if state.get("generated_content") else "failed"

    # Node: write the generated content to disk via the shared noteWriter
    # (moved from the old notesTool -- same sanitization/disambiguation logic).
    def save_note_node(state: ActionsSubgraphState) -> dict:

        try:
            saved_path = save_note(state["generated_content"], state["generated_file_name"])
        except Exception as e:
            return {"tool_result": {"output": f"Actions tool error: failed to save note ({e})", "relay_verbatim": False}}

        return {"tool_result": {"output": f"Note saved to {saved_path}.", "relay_verbatim": False}}

    graph.add_node("extract_request", extract_request)
    graph.add_node("generate_content", generate_content)
    graph.add_node("save_note", save_note_node)

    graph.add_edge(START, "extract_request")
    graph.add_conditional_edges(
        "extract_request",
        route_after_extract,
        {
            "has_request": "generate_content",
            "empty_request": END,
        },
    )
    graph.add_conditional_edges(
        "generate_content",
        route_after_generate,
        {
            "success": "save_note",
            "failed": END,
        },
    )
    graph.add_edge("save_note", END)

    return graph.compile()
