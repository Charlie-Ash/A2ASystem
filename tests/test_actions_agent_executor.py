# Drives ActionsAgentExecutor through the real a2a-sdk request-handling stack
# (DefaultRequestHandler -> TaskManager -> ResultAggregator ->
# InMemoryTaskStore), using build_fake_actions_subgraph() in place of the
# real GPU-backed subgraph -- a2a-sdk itself needs no GPU, so this is real
# (non-mocked) framework coverage of the executor's wiring, just with a fake
# Actions subgraph underneath it. See fake_dependencies.py for why the fake
# is an actually-compiled StateGraph rather than a bare stub function.
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

from tools.actionsTool.a2a.agent_executor import ActionsAgentExecutor
from tools.actionsTool.actions.state import ActionsSubgraphState

from fake_dependencies import build_fake_actions_subgraph


def _send_params(text: str) -> MessageSendParams:

    return MessageSendParams(
        message=Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=text))],
            message_id=str(uuid.uuid4()),
        )
    )


def _build_handler(canned_output="actions-output", relay_verbatim=False, received_args_sink=None):

    subgraph = build_fake_actions_subgraph(
        canned_output=canned_output,
        relay_verbatim=relay_verbatim,
        received_args_sink=received_args_sink,
    )
    executor = ActionsAgentExecutor(subgraph)
    return DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())


async def test_message_send_completes_and_returns_the_subgraphs_answer():

    handler = _build_handler(canned_output="Note saved to data/notes/pete_astronomy_notes.txt.")
    result = await handler.on_message_send(_send_params("remember Pete likes astronomy"))

    assert result.status.state == TaskState.completed
    assert len(result.artifacts) == 1
    assert result.artifacts[0].parts[0].root.text == "Note saved to data/notes/pete_astronomy_notes.txt."


async def test_message_send_passes_the_users_text_through_as_the_actions_request():

    received_args_sink = {"args": None}
    handler = _build_handler(received_args_sink=received_args_sink)

    await handler.on_message_send(_send_params("remember Pete likes astronomy"))

    assert received_args_sink["args"] == {"request": "remember Pete likes astronomy"}


async def test_two_calls_are_independent_of_each_other():
    # No checkpointer is attached when the subgraph is invoked as a
    # standalone top-level graph (see graph.py's build_actions_subgraph()
    # docstring), so two separate calls should each get their own fresh
    # completed task rather than colliding on shared state.

    handler = _build_handler(canned_output="saved")

    result_1 = await handler.on_message_send(_send_params("note one"))
    result_2 = await handler.on_message_send(_send_params("note two"))

    assert result_1.id != result_2.id
    assert result_1.status.state == TaskState.completed
    assert result_2.status.state == TaskState.completed


async def test_subgraph_exception_fails_the_task_instead_of_raising():

    def broken_run(state: ActionsSubgraphState) -> dict:
        raise RuntimeError("boom")

    sub = StateGraph(ActionsSubgraphState)
    sub.add_node("broken_run", broken_run)
    sub.add_edge(START, "broken_run")
    sub.add_edge("broken_run", END)
    broken_subgraph = sub.compile()

    executor = ActionsAgentExecutor(broken_subgraph)
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

    executor = ActionsAgentExecutor(build_fake_actions_subgraph())
    context = RequestContext(request=_send_params("cancel me"))
    queue = EventQueue()

    try:
        await executor.cancel(context, queue)
        assert False, "expected cancel() to raise"
    except ServerError:
        pass
