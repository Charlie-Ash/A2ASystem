# Bridges the A2A protocol server to the RAG tool's own compiled subgraph
# (see tools/ragTool/rag/graph.py). Each incoming A2A message becomes exactly
# one ToolCall(tool="rag", action="run", args={"query": ...}) fed into the
# subgraph -- the same ToolCall the orchestrator already builds today before
# calling this subgraph in-process. This executor is just a second front
# door onto the same subgraph, run here as a standalone top-level graph
# (tools/ragTool/a2a/a2a_server.py) instead of a node nested under the
# orchestrator's own graph, so it has no checkpointer/cross-call memory of
# its own.
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TextPart, UnsupportedOperationError
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from schemas.tool_call import ToolCall


class RAGAgentExecutor(AgentExecutor):

    def __init__(self, subgraph):

        # subgraph: the compiled graph from build_rag_subgraph() -- either
        # the real RAGTool().subgraph (a2a_server.py) or a fake one built by
        # tests/fake_dependencies.py's build_fake_rag_subgraph() (tests).
        self.subgraph = subgraph

    # Runs one turn of the RAG subgraph for an incoming A2A message and
    # reports the result back as a completed (or failed) task.
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:

        query = context.get_user_input()

        task = context.current_task or new_task(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        # One cheap, honest progress signal before the (potentially slow,
        # vLLM-backed) invoke below -- confirmed this can't turn a blocking
        # message/send call into a non-blocking one, so it costs nothing and
        # is what a client watching task state would expect to see instead
        # of the task sitting silently in "submitted".
        await updater.start_work()

        tool_call = ToolCall(tool="rag", action="run", args={"query": query})

        try:
            result = await self.subgraph.ainvoke({"tool_call": tool_call})
        except Exception as e:
            await updater.failed(
                new_agent_text_message(f"RAG agent failed: {e}", task.context_id, task.id)
            )
            return

        # tool_result's relay_verbatim isn't surfaced in task state here --
        # even a ToolResult with relay_verbatim=False (e.g. the subgraph's
        # own "no query provided" error text) still completes the task, same
        # as how the orchestrator's generate_response node already treats
        # this ToolResult as content to compose around rather than a failure.
        tool_result = result["tool_result"]

        await updater.add_artifact(
            [Part(root=TextPart(text=tool_result["output"]))],
            name="rag_answer",
        )
        await updater.complete()

    # No cancellation support: a single subgraph.ainvoke() call has no
    # partial-progress state worth preserving.
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:

        raise ServerError(error=UnsupportedOperationError())
