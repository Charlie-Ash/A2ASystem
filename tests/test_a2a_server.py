# Smoke tests for the standalone RAG agent's server wiring -- build_agent_card()
# is a plain function with no GPU/Qdrant side effects (unlike main(), which
# constructs the real RAGTool()), so it and the FastAPI app it feeds into can
# be exercised directly here without a GPU. Confirms the agent card is valid
# and actually reachable at the well-known discovery URL a real A2A client
# would GET first.
from fastapi.testclient import TestClient

from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from tools.ragTool.a2a.a2a_server import build_agent_card
from tools.ragTool.a2a.agent_executor import RAGAgentExecutor

from fake_dependencies import build_fake_rag_subgraph


def _build_app():

    agent_card = build_agent_card(host="localhost", port=8001)
    request_handler = DefaultRequestHandler(
        agent_executor=RAGAgentExecutor(build_fake_rag_subgraph()),
        task_store=InMemoryTaskStore(),
    )
    application = A2AFastAPIApplication(agent_card=agent_card, http_handler=request_handler)
    return application.build()


def test_agent_card_advertises_the_rag_skill():

    card = build_agent_card(host="localhost", port=8001)

    assert card.name == "RAG Document Agent"
    assert [skill.id for skill in card.skills] == ["answer_from_company_documents"]
    assert card.capabilities.streaming is False


def test_agent_card_is_served_at_the_well_known_discovery_url():

    client = TestClient(_build_app())

    response = client.get("/.well-known/agent-card.json")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "RAG Document Agent"
    assert body["skills"][0]["id"] == "answer_from_company_documents"
