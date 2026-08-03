# Runs the compiled graph end-to-end with fake LLM/tool dependencies (no
# GPU), checking that each tool branch is actually reached and that the two
# rule-based ToolCall.args overrides (RAG always gets the exact user
# question; a note never saves with blank content) still fire correctly now
# that they live inside graph.py's decide_tool node instead of
# orchestrator.py.
import pytest

from schemas.tool_call import ToolCall
from orchestrator.graph import build_graph

from fake_dependencies import FakeOrchestratorLLM, FakeToolRouter


def _thread_config(thread_id="test-thread"):
    return {"configurable": {"thread_id": thread_id}}


@pytest.mark.parametrize("tool_name", ["default", "rag", "note"])
def test_each_tool_branch_is_reached_and_produces_a_final_response(tool_name):

    llm = FakeOrchestratorLLM(next_tool_call=ToolCall(tool=tool_name, action="run", args={}))
    tool_router = FakeToolRouter()
    graph = build_graph(llm, tool_router)

    result = graph.invoke({"user_message": "hello there"}, config=_thread_config())

    # The matching fake tool actually ran (and no other one did).
    assert tool_router.tools[tool_name].received_args is not None
    for other_name, other_tool in tool_router.tools.items():
        if other_name != tool_name:
            assert other_tool.received_args is None

    # generate_response and orchestrator_mem_update both ran, downstream of
    # whichever tool node fired.
    assert result["final_response"] == f"canned reply about {tool_name}-output"
    assert len(llm.orchestrator_mem_update_calls) == 1


def test_rag_branch_always_receives_the_users_exact_message_as_query():

    # tool_decision "forgot" to fill in a query -- decide_tool's rule-based
    # override should fill it in from the raw user message regardless.
    llm = FakeOrchestratorLLM(next_tool_call=ToolCall(tool="rag", action="run", args={}))
    tool_router = FakeToolRouter()
    graph = build_graph(llm, tool_router)

    graph.invoke({"user_message": "what is Pete's favorite subject?"}, config=_thread_config())

    assert tool_router.tools["rag"].received_args == {"query": "what is Pete's favorite subject?"}


def test_note_branch_falls_back_to_user_message_when_content_is_missing():

    llm = FakeOrchestratorLLM(next_tool_call=ToolCall(tool="note", action="run", args={}))
    tool_router = FakeToolRouter()
    graph = build_graph(llm, tool_router)

    graph.invoke({"user_message": "remember that Pete likes astronomy"}, config=_thread_config())

    assert tool_router.tools["note"].received_args == {"content": "remember that Pete likes astronomy"}


def test_note_branch_keeps_llm_provided_content_when_present():

    llm = FakeOrchestratorLLM(
        next_tool_call=ToolCall(
            tool="note", action="run", args={"content": "Pete likes astronomy", "file_name": "pete"}
        )
    )
    tool_router = FakeToolRouter()
    graph = build_graph(llm, tool_router)

    graph.invoke({"user_message": "remember Pete likes astronomy"}, config=_thread_config())

    assert tool_router.tools["note"].received_args == {
        "content": "Pete likes astronomy",
        "file_name": "pete",
    }


def test_conversation_history_persists_across_turns_with_same_thread_id():

    # Regression coverage for the MemorySaver checkpointer itself: two
    # invoke() calls against the same thread_id should let the second turn's
    # prompts see the first turn's HumanMessage/AIMessage as history, since
    # that's the entire point of wiring the checkpointer up (see graph.py).
    llm = FakeOrchestratorLLM(next_tool_call=ToolCall(tool="default", action="run", args={}))
    tool_router = FakeToolRouter()
    graph = build_graph(llm, tool_router)
    config = _thread_config("persistent-thread")

    graph.invoke({"user_message": "first message"}, config=config)
    graph.invoke({"user_message": "second message"}, config=config)

    assert len(llm.tool_decision_calls) == 2
    _, first_turn_history = llm.tool_decision_calls[0]
    second_user_message, second_turn_history = llm.tool_decision_calls[1]

    assert second_user_message == "second message"
    assert first_turn_history == []
    assert [m.content for m in second_turn_history] == ["first message", "canned reply about default-output"]


def test_independent_thread_ids_do_not_share_history():

    llm = FakeOrchestratorLLM(next_tool_call=ToolCall(tool="default", action="run", args={}))
    tool_router = FakeToolRouter()
    graph = build_graph(llm, tool_router)

    graph.invoke({"user_message": "hello from thread A"}, config=_thread_config("thread-a"))
    graph.invoke({"user_message": "hello from thread B"}, config=_thread_config("thread-b"))

    _, second_call_history = llm.tool_decision_calls[1]
    assert second_call_history == []
