# Shared contract every tool's run() must satisfy. This used to be an
# unwritten convention (each tool just returned a bare string, and the RAG
# tool smuggled a formatting instruction inside that string for prompts.py to
# string-split back out). Making it an explicit typed structure means the
# orchestrator can hand tool output around the graph without caring which
# tool produced it, or unpicking it, which is what lets a tool later become a
# real subgraph/agent without changing anything downstream.
from typing import Protocol, TypedDict


class ToolResult(TypedDict):

    # The tool's actual output text.
    output: str

    # True if "output" is already a complete, user-ready answer that should be
    # shown as-is (e.g. the RAG tool's generated answer); False if it's raw
    # material (a status message, an error) that the orchestrator's
    # response-generation step should compose into a reply.
    relay_verbatim: bool


class Tool(Protocol):
    """Structural contract for anything registered in ToolRouter.

    A class satisfies this just by having a matching run() method --
    it does not need to inherit from anything.
    """

    def run(self, tool_args: dict) -> ToolResult: ...
