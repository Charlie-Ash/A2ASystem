# Bridges the A2A protocol server to the Actions tool's own compiled
# subgraph (see tools/actionsTool/actions/graph.py). Each incoming A2A
# message becomes exactly one ToolCall(tool="actions", action="run",
# args={"request": ...}), built here from the raw message text -- the
# orchestrator itself no longer constructs a ToolCall or touches this
# subgraph at all; it only ever reaches this agent over the network (see
# orchestrator/agents/remote_agent.py). This executor runs the subgraph as a
# standalone top-level graph (tools/actionsTool/a2a/a2a_server.py), so it has
# no checkpointer/cross-call memory of its own.
#
# Cross-process chat history: the orchestrator's own MemorySaver-backed
# OrchestratorState.messages can't be shared with this process directly (no
# Python object crosses a real network boundary), so instead the orchestrator
# attaches a recent slice of history to the incoming Message's `metadata`
# field explicitly (see orchestrator/agents/remote_agent.py's
# call_remote_agent). This executor reads that back and reconstructs it into
# the subgraph's expected "messages" state key, current turn appended last --
# same convention the orchestrator's own decide_tool uses. RAG's standalone
# server never depended on chat history in the first place, so it has no
# equivalent need.
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TextPart, UnsupportedOperationError
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from langchain_core.messages import HumanMessage

from chat_history import chat_messages_to_history
from schemas.tool_call import ToolCall


class ActionsAgentExecutor(AgentExecutor):

    def __init__(self, subgraph):

        # subgraph: the compiled graph from build_actions_subgraph() --
        # either the real ActionsTool().subgraph (a2a_server.py) or a fake
        # one built by tests/fake_dependencies.py's build_fake_actions_subgraph() (tests).
        self.subgraph = subgraph

    # Runs one turn of the Actions subgraph for an incoming A2A message and
    # reports the result back as a completed (or failed) task.
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:

        request = context.get_user_input()

        task = context.current_task or new_task(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        # One cheap, honest progress signal before the (potentially slow,
        # vLLM-backed) invoke below -- same reasoning as RAGAgentExecutor.
        await updater.start_work()

        tool_call = ToolCall(tool="actions", action="run", args={"request": request})

        # Reconstruct any history the orchestrator attached to this message
        # (see module comment above); [] if none was sent (e.g. a direct
        # A2A client with no conversation to relay, or the very first turn).
        wire_history = (context.message.metadata or {}).get("history", []) if context.message else []
        history_messages = chat_messages_to_history(wire_history)

        try:
            result = await self.subgraph.ainvoke({
                "tool_call": tool_call,
                "messages": [*history_messages, HumanMessage(content=request)],
            })
        except Exception as e:
            await updater.failed(
                new_agent_text_message(f"Actions agent failed: {e}", task.context_id, task.id)
            )
            return

        # tool_result's relay_verbatim isn't surfaced in task state here --
        # even a ToolResult with relay_verbatim=False (e.g. an empty-request
        # or generation-failure error) still completes the task, same as how
        # the orchestrator's generate_response node already treats this
        # ToolResult as content to compose around rather than a failure.
        tool_result = result["tool_result"]

        await updater.add_artifact(
            [Part(root=TextPart(text=tool_result["output"]))],
            name="actions_result",
        )
        await updater.complete()

    # No cancellation support: a single subgraph.ainvoke() call has no
    # partial-progress state worth preserving.
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:

        raise ServerError(error=UnsupportedOperationError())
