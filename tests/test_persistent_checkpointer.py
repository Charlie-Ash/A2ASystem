# Proves cross-*process* durability, not just cross-turn durability within
# one in-RAM MemorySaver: builds a graph against a real (temp-file) sqlite
# database, runs a turn, closes that connection, then builds a SECOND,
# independent graph instance against the SAME file with a fresh
# AsyncSqliteSaver -- simulating an orchestrator process restart -- and
# confirms the second instance resumes the first's history. GPU-free, reuses
# fake_dependencies.py's existing fakes; the one test in the suite that
# touches a real sqlite file rather than pure in-memory fakes.
#
# No explicit checkpointer.setup() call: AsyncSqliteSaver's own docstring
# says it's called automatically by every read/write method as needed and
# "should not be called directly by the user" -- graph.ainvoke() below
# triggers it on first use.
import tempfile
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from schemas.tool_call import ToolCall
from orchestrator.pipeline.graph import build_graph

from fake_dependencies import FakeOrchestratorLLM, FakeToolRouter


async def test_second_graph_instance_against_the_same_sqlite_file_resumes_history():

    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "checkpoints.db")
        thread_config = {"configurable": {"thread_id": "restart-thread"}}

        llm_1 = FakeOrchestratorLLM(next_tool_call=ToolCall(tool="default", action="run", args={}))
        tool_router_1 = await FakeToolRouter.create()
        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer_1:
            graph_1 = build_graph(llm_1, tool_router_1, checkpointer=checkpointer_1)
            await graph_1.ainvoke({"user_message": "first message"}, config=thread_config)
        # checkpointer_1's connection is now closed -- simulates process exit.

        llm_2 = FakeOrchestratorLLM(next_tool_call=ToolCall(tool="default", action="run", args={}))
        tool_router_2 = await FakeToolRouter.create()
        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer_2:
            graph_2 = build_graph(llm_2, tool_router_2, checkpointer=checkpointer_2)
            await graph_2.ainvoke({"user_message": "second message"}, config=thread_config)

        assert len(llm_2.tool_decision_calls) == 1
        _, history, _ = llm_2.tool_decision_calls[0]
        assert [m.content for m in history] == ["first message", "canned reply about default-output"]
