# Standalone entrypoint: starts the RAG tool as its own A2A protocol server
# process (own FastAPI/uvicorn, own vLLM engine, own embedded Qdrant index --
# see tools/ragTool/rag/tool_class.py). Run as
# `python -m tools.ragTool.a2a.a2a_server` from inside src/, the same way
# main.py is run today -- NOT `python a2a_server.py` directly, which would
# put the wrong directory on sys.path and break this project's
# absolute-import convention.
#
# Don't run this alongside main.py right now: both would try to open the
# same on-disk embedded Qdrant path (config.py's QDRANT_DB_PATH), and
# embedded Qdrant only allows one process to hold that file lock at a time.
import uvicorn

from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from tools.ragTool.a2a.agent_executor import RAGAgentExecutor
from tools.ragTool.config import RAG_A2A_HOST, RAG_A2A_PORT


# Builds this agent's card: the JSON document A2A clients GET at
# /.well-known/agent-card.json to discover what this agent can do, before
# ever sending it work. Kept as its own function (not inlined in main())
# so it can be constructed and asserted on in tests without needing
# RAGTool()'s GPU/Qdrant side effects.
def build_agent_card(host: str, port: int) -> AgentCard:

    skill = AgentSkill(
        id="answer_from_company_documents",
        name="Answer questions from company documents",
        description=(
            "Answers a natural-language question using retrieval-augmented "
            "generation over the company's ingested document collection. "
            "Retrieves the most relevant passages and generates a grounded "
            "answer from them."
        ),
        tags=["rag", "documents", "question-answering"],
        examples=[
            "What is our refund policy?",
            "Summarize the onboarding checklist for new hires.",
        ],
    )

    return AgentCard(
        name="RAG Document Agent",
        description=(
            "Standalone agent that answers questions from the company's "
            "document collection via retrieval-augmented generation."
        ),
        url=f"http://{host}:{port}/",
        version="0.1.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        # Deliberately honest: the executor never checks push-notification
        # config and only emits one interim "working" status, so don't
        # advertise more than that.
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        skills=[skill],
    )


def main():

    # Only place in this process that touches GPU/Qdrant -- builds the
    # vector index + vLLM engine exactly as the orchestrator's in-process
    # RAGTool() does today, just now happening in its own process.
    from tools.ragTool.rag.tool_class import RAGTool

    rag_tool = RAGTool()

    agent_card = build_agent_card(RAG_A2A_HOST, RAG_A2A_PORT)

    request_handler = DefaultRequestHandler(
        agent_executor=RAGAgentExecutor(rag_tool.subgraph),
        task_store=InMemoryTaskStore(),
    )

    app = A2AFastAPIApplication(agent_card=agent_card, http_handler=request_handler)

    # Blocks forever, same shape as main.py's blocking input() loop -- this
    # call never returns while the server is up.
    uvicorn.run(app.build(), host=RAG_A2A_HOST, port=RAG_A2A_PORT)


if __name__ == "__main__":
    main()
