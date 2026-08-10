# Standalone entrypoint: starts the Actions tool as its own A2A protocol
# server process (own FastAPI/uvicorn, own vLLM engine -- see
# tools/actionsTool/actions/tool_class.py). Run as
# `python -m tools.actionsTool.a2a.a2a_server` from inside src/, the same way
# main.py and RAG's a2a_server.py are run today -- NOT `python a2a_server.py`
# directly, which would put the wrong directory on sys.path and break this
# project's absolute-import convention.
import uvicorn

from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from tools.actionsTool.a2a.agent_executor import ActionsAgentExecutor
from tools.actionsTool.config import ACTIONS_A2A_HOST, ACTIONS_A2A_PORT


# Builds this agent's card: the JSON document A2A clients GET at
# /.well-known/agent-card.json to discover what this agent can do, before
# ever sending it work. Kept as its own function (not inlined in main())
# so it can be constructed and asserted on in tests without needing
# ActionsTool()'s GPU side effects.
def build_agent_card(host: str, port: int) -> AgentCard:

    skill = AgentSkill(
        id="save_note",
        name="Save a note",
        description=(
            "Writes down information the user asks to remember, as a short "
            "note saved to a text file. Generates the note's content itself "
            "from the request (optionally using recent conversation history "
            "to resolve references to earlier turns) and picks a short, "
            "filesystem-safe file name for it."
        ),
        tags=["actions", "notes", "write"],
        examples=[
            "Remember Pete likes astronomy",
            "Note down your previous answer to this question",
        ],
    )

    return AgentCard(
        name="Actions Agent",
        description=(
            "Standalone agent that writes user-provided information to a "
            "note file, generating the note's content itself."
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

    # Only place in this process that touches GPU -- builds the vLLM engine
    # exactly as the orchestrator's in-process ActionsTool() does today, just
    # now happening in its own process.
    from tools.actionsTool.actions.tool_class import ActionsTool

    actions_tool = ActionsTool()

    agent_card = build_agent_card(ACTIONS_A2A_HOST, ACTIONS_A2A_PORT)

    request_handler = DefaultRequestHandler(
        agent_executor=ActionsAgentExecutor(actions_tool.subgraph),
        task_store=InMemoryTaskStore(),
    )

    app = A2AFastAPIApplication(agent_card=agent_card, http_handler=request_handler)

    # Blocks forever, same shape as main.py's blocking input() loop -- this
    # call never returns while the server is up.
    uvicorn.run(app.build(), host=ACTIONS_A2A_HOST, port=ACTIONS_A2A_PORT)


if __name__ == "__main__":
    main()
