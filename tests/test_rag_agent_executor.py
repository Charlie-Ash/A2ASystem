# Drives RAGAgentExecutor through the real a2a-sdk request-handling stack
# (DefaultRequestHandler -> TaskManager -> ResultAggregator ->
# InMemoryTaskStore), using build_fake_rag_subgraph() in place of the real
# GPU/Qdrant-backed subgraph -- a2a-sdk itself needs no GPU, so this is real
# (non-mocked) framework coverage of the executor's wiring, just with a fake
# RAG subgraph underneath it. See fake_dependencies.py for why the fake is
# an actually-compiled StateGraph rather than a bare stub function.
import uuid

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    Message,
    MessageSendParams,
    Part,
    Role,
    TaskState,
    TextPart,
)
from a2a.utils.errors import ServerError

from langgraph.graph import StateGraph, START, END

from tools.ragTool.a2a.agent_executor import RAGAgentExecutor
from tools.ragTool.rag.state import RAGSubgraphState

from fake_dependencies import build_fake_rag_subgraph


def _send_params(text: str) -> MessageSendParams:

    return MessageSendParams(
        message=Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=text))],
            message_id=str(uuid.uuid4()),
        )
    )


def _build_handler(canned_output="rag-output", relay_verbatim=True, received_args_sink=None):

    subgraph = build_fake_rag_subgraph(
        canned_output=canned_output,
        relay_verbatim=relay_verbatim,
        received_args_sink=received_args_sink,
    )
    executor = RAGAgentExecutor(subgraph)
    return DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())


async def test_message_send_completes_and_returns_the_subgraphs_answer():

    handler = _build_handler(canned_output="the refund policy is 30 days")
    result = await handler.on_message_send(_send_params("what is the refund policy?"))

    assert result.status.state == TaskState.completed
    assert len(result.artifacts) == 1
    assert result.artifacts[0].parts[0].root.text == "the refund policy is 30 days"


async def test_message_send_passes_the_users_text_through_as_the_rag_query():

    received_args_sink = {"args": None}
    handler = _build_handler(received_args_sink=received_args_sink)

    await handler.on_message_send(_send_params("what is the refund policy?"))

    assert received_args_sink["args"] == {"query": "what is the refund policy?"}


async def test_two_calls_are_independent_of_each_other():
    # No checkpointer is attached when the subgraph is invoked as a
    # standalone top-level graph (see graph.py's build_rag_subgraph()
    # docstring), so two separate calls should each get their own fresh
    # completed task rather than colliding on shared state.

    handler = _build_handler(canned_output="answer")

    result_1 = await handler.on_message_send(_send_params("q1"))
    result_2 = await handler.on_message_send(_send_params("q2"))

    assert result_1.id != result_2.id
    assert result_1.status.state == TaskState.completed
    assert result_2.status.state == TaskState.completed


async def test_subgraph_exception_fails_the_task_instead_of_raising():

    def broken_run(state: RAGSubgraphState) -> dict:
        raise RuntimeError("boom")

    sub = StateGraph(RAGSubgraphState)
    sub.add_node("broken_run", broken_run)
    sub.add_edge(START, "broken_run")
    sub.add_edge("broken_run", END)
    broken_subgraph = sub.compile()

    executor = RAGAgentExecutor(broken_subgraph)
    context = RequestContext(request=_send_params("this will blow up"))
    queue = EventQueue()

    await executor.execute(context, queue)

    # Calling execute() directly (bypassing DefaultRequestHandler's
    # consumer, which is what normally closes the queue on a final event)
    # means the queue is never auto-closed here -- drain events until the
    # TaskStatusUpdateEvent marked final=True instead of waiting on
    # is_closed(). Events: task creation, the "working" status, then the
    # terminal "failed" status.
    final_status = None
    for _ in range(10):
        event = await queue.dequeue_event()
        queue.task_done()
        if hasattr(event, "final") and event.final:
            final_status = event
            break

    assert final_status is not None
    assert final_status.status.state == TaskState.failed


async def test_cancel_is_unsupported():

    executor = RAGAgentExecutor(build_fake_rag_subgraph())
    context = RequestContext(request=_send_params("cancel me"))
    queue = EventQueue()

    try:
        await executor.cancel(context, queue)
        assert False, "expected cancel() to raise"
    except ServerError:
        pass
