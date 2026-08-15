# Runs the compiled graph end-to-end with a fake LLM and a fake-subgraph-backed
# real A2A server underneath (no GPU, no real network -- see
# fake_dependencies.py), checking that each tool branch is actually reached
# and that the remote agents always receive the user's exact message as A2A
# message text, regardless of whatever (if anything) the LLM's tool_call.args
# contained -- tool_call.args is no longer read for remote tools at all now
# that decide_tool's old per-tool override rules are gone (see graph.py).
import pytest

from schemas.tool_call import ToolCall
from orchestrator.pipeline.graph import build_graph

from fake_dependencies import FakeOrchestratorLLM, FakeToolRouter


def _thread_config(thread_id="test-thread"):
    return {"configurable": {"thread_id": thread_id}}


@pytest.mark.parametrize("tool_name", ["default", "rag", "actions"])
async def test_each_tool_branch_is_reached_and_produces_a_final_response(tool_name):

    llm = FakeOrchestratorLLM(next_tool_call=ToolCall(tool=tool_name, action="run", args={}))
    tool_router = await FakeToolRouter.create()
    graph = build_graph(llm, tool_router)

    result = await graph.ainvoke({"user_message": "hello there"}, config=_thread_config())

    # The matching fake tool/agent actually ran (and no other one did).
    # "rag"/"actions" aren't in tool_router.tools anymore (see
    # FakeToolRouter) -- they're tracked separately via
    # rag_received_args/actions_received_args, populated by the real
    # RAGAgentExecutor/ActionsAgentExecutor rebuilding a ToolCall server-side
    # from the A2A message text (see tools/*/a2a/agent_executor.py).
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


async def test_rag_branch_always_receives_the_users_exact_message_as_query():

    # Even though tool_decision didn't fill in a query, the remote RAG agent
    # still gets the user's exact message -- run_remote_tool sends
    # state["user_message"] as the A2A message text unconditionally (see
    # graph.py's make_remote_tool_node), and RAGAgentExecutor reconstructs
    # {"query": <that text>} server-side.
    llm = FakeOrchestratorLLM(next_tool_call=ToolCall(tool="rag", action="run", args={}))
    tool_router = await FakeToolRouter.create()
    graph = build_graph(llm, tool_router)

    await graph.ainvoke({"user_message": "what is Pete's favorite subject?"}, config=_thread_config())

    assert tool_router.rag_received_args["args"] == {"query": "what is Pete's favorite subject?"}


async def test_actions_branch_always_receives_the_users_exact_message_as_request():

    # Same reasoning as the RAG test above: whatever the LLM put in
    # tool_call.args (here, a deliberately wrong "request") is ignored --
    # the Actions agent always receives the user's exact message as A2A text.
    llm = FakeOrchestratorLLM(
        next_tool_call=ToolCall(tool="actions", action="run", args={"request": "something else entirely"})
    )
    tool_router = await FakeToolRouter.create()
    graph = build_graph(llm, tool_router)

    await graph.ainvoke({"user_message": "remember Pete likes astronomy"}, config=_thread_config())

    assert tool_router.actions_received_args["args"] == {"request": "remember Pete likes astronomy"}


async def test_actions_branch_receives_chat_history_over_a2a_metadata():
    # Once Actions is called over a real A2A boundary, there's no shared
    # Python-level MemorySaver/state to inherit chat history from directly --
    # so instead the orchestrator attaches a slice of state["messages"] to
    # the outgoing A2A message's metadata field explicitly (see
    # orchestrator/agents/remote_agent.py's call_remote_agent), and
    # ActionsAgentExecutor reconstructs it back into "messages" server-side
    # (see tools/actionsTool/a2a/agent_executor.py). This is what makes "note
    # down your previous answer" work again through the Actions agent.
    llm = FakeOrchestratorLLM(next_tool_call=ToolCall(tool="default", action="run", args={}))
    tool_router = await FakeToolRouter.create()
    graph = build_graph(llm, tool_router)
    config = _thread_config("actions-history-thread")

    await graph.ainvoke({"user_message": "first message"}, config=config)

    llm.next_tool_call = ToolCall(tool="actions", action="run", args={})
    await graph.ainvoke({"user_message": "note down your previous answer"}, config=config)

    messages = tool_router.actions_received_messages["messages"]
    assert [(type(m).__name__, m.content) for m in messages] == [
        ("HumanMessage", "first message"),
        ("AIMessage", "canned reply about default-output"),
        ("HumanMessage", "note down your previous answer"),
    ]


async def test_conversation_history_persists_across_turns_with_same_thread_id():

    # Regression coverage for the MemorySaver checkpointer itself: two
    # invoke() calls against the same thread_id should let the second turn's
    # prompts see the first turn's HumanMessage/AIMessage as history, since
    # that's the entire point of wiring the checkpointer up (see graph.py).
    # This is about the orchestrator's own state, not any tool/agent's --
    # unaffected by the A2A rework.
    llm = FakeOrchestratorLLM(next_tool_call=ToolCall(tool="default", action="run", args={}))
    tool_router = await FakeToolRouter.create()
    graph = build_graph(llm, tool_router)
    config = _thread_config("persistent-thread")

    await graph.ainvoke({"user_message": "first message"}, config=config)
    await graph.ainvoke({"user_message": "second message"}, config=config)

    assert len(llm.tool_decision_calls) == 2
    _, first_turn_history, _ = llm.tool_decision_calls[0]
    second_user_message, second_turn_history, _ = llm.tool_decision_calls[1]

    assert second_user_message == "second message"
    assert first_turn_history == []
    assert [m.content for m in second_turn_history] == ["first message", "canned reply about default-output"]


async def test_independent_thread_ids_do_not_share_history():

    llm = FakeOrchestratorLLM(next_tool_call=ToolCall(tool="default", action="run", args={}))
    tool_router = await FakeToolRouter.create()
    graph = build_graph(llm, tool_router)

    await graph.ainvoke({"user_message": "hello from thread A"}, config=_thread_config("thread-a"))
    await graph.ainvoke({"user_message": "hello from thread B"}, config=_thread_config("thread-b"))

    _, second_call_history, _ = llm.tool_decision_calls[1]
    assert second_call_history == []
