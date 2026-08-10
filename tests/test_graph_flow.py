# Runs the compiled graph end-to-end with fake LLM/tool dependencies (no
# GPU), checking that each tool branch is actually reached and that the two
# rule-based ToolCall.args overrides (RAG always gets the exact user
# question; Actions always gets the exact user message as its request) still
# fire correctly now that they live inside graph.py's decide_tool node
# instead of orchestrator.py.
import pytest

from schemas.tool_call import ToolCall
from orchestrator.graph import build_graph

from fake_dependencies import FakeOrchestratorLLM, FakeToolRouter


def _thread_config(thread_id="test-thread"):
    return {"configurable": {"thread_id": thread_id}}


@pytest.mark.parametrize("tool_name", ["default", "rag", "actions"])
def test_each_tool_branch_is_reached_and_produces_a_final_response(tool_name):

    llm = FakeOrchestratorLLM(next_tool_call=ToolCall(tool=tool_name, action="run", args={}))
    tool_router = FakeToolRouter()
    graph = build_graph(llm, tool_router)

    result = graph.invoke({"user_message": "hello there"}, config=_thread_config())

    # The matching fake tool/subgraph actually ran (and no other one did).
    # "rag"/"actions" aren't in tool_router.tools anymore (see
    # FakeToolRouter) -- they're tracked separately via
    # rag_received_args/actions_received_args.
    if tool_name == "rag":
        assert tool_router.rag_received_args["args"] is not None
        assert tool_router.actions_received_args["args"] is None
    elif tool_name == "actions":
        assert tool_router.actions_received_args["args"] is not None
        assert tool_router.rag_received_args["args"] is None
    else:
        assert tool_router.tools[tool_name].received_args is not None
        assert tool_router.rag_received_args["args"] is None
        assert tool_router.actions_received_args["args"] is None

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

    assert tool_router.rag_received_args["args"] == {"query": "what is Pete's favorite subject?"}


def test_actions_branch_always_receives_the_users_exact_message_as_request():

    # Content generation now happens inside the Actions subgraph itself, not
    # in decide_tool -- so decide_tool's rule-based override always sets
    # "request" to the raw user message, regardless of whatever (if
    # anything) the LLM put in args, same shape as RAG's "query" override.
    llm = FakeOrchestratorLLM(
        next_tool_call=ToolCall(tool="actions", action="run", args={"request": "something else entirely"})
    )
    tool_router = FakeToolRouter()
    graph = build_graph(llm, tool_router)

    graph.invoke({"user_message": "remember Pete likes astronomy"}, config=_thread_config())

    assert tool_router.actions_received_args["args"] == {"request": "remember Pete likes astronomy"}


def test_actions_branch_receives_chat_history_from_the_parent_graph():

    # Regression coverage for ActionsSubgraphState's shared "messages" key
    # (see tools/actionsTool/actions/state.py): without it, the Actions
    # agent's content-generation node would have no way to resolve
    # references like "note down your previous answer to this question".
    llm = FakeOrchestratorLLM(next_tool_call=ToolCall(tool="default", action="run", args={}))
    tool_router = FakeToolRouter()
    graph = build_graph(llm, tool_router)
    config = _thread_config("actions-history-thread")

    graph.invoke({"user_message": "first message"}, config=config)

    llm.next_tool_call = ToolCall(tool="actions", action="run", args={})
    graph.invoke({"user_message": "note down your previous answer"}, config=config)

    received_messages = tool_router.actions_received_messages["messages"]
    assert received_messages is not None
    # The fake node records the raw state["messages"] it's handed as a
    # subgraph -- this is a structural check that the shared "messages" key
    # actually crosses the subgraph boundary with the checkpointed history,
    # not a check of generate_content's own [:-1] "drop the current turn"
    # slicing (that's real-node logic, exercised only on the GPU machine,
    # same as generate_answer's real logic isn't unit-tested here either).
    # So this includes the first turn's HumanMessage + AIMessage *and* this
    # turn's own just-appended HumanMessage.
    assert [m.content for m in received_messages] == [
        "first message",
        "canned reply about default-output",
        "note down your previous answer",
    ]


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
