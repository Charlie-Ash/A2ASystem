# Actions (note-writing) operations via vLLM
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from tools.gpu_utils import check_gpu_memory
from tools.actionsTool.actions.graph import build_actions_subgraph
from tools.actionsTool.actions.note_schema import NOTE_SCHEMA
from tools.actionsTool.config import LLM_MODEL, GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN


class ActionsTool():

    def __init__(self):

        print("Actions tool initialized.")

        print("Loading Actions LLM...")
        check_gpu_memory(GPU_MEMORY_UTILIZATION)

        try:
            self.llm = LLM(
                model=LLM_MODEL,
                trust_remote_code=True,  # required for Gemma's custom architecture code
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                max_model_len=MAX_MODEL_LEN
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load LLM '{LLM_MODEL}': {e}") from e

        self.sampling_params = SamplingParams(
            temperature=0.3,  # some randomness for natural note phrasing, but still fairly literal
            top_p=0.9,  # top_p; nucleus sampling
            max_tokens=300,  # notes are meant to stay short
            repetition_penalty=1.1,  # Penalty to apply if tokens continue repeating.
            structured_outputs=StructuredOutputsParams(json=NOTE_SCHEMA)  # Forces output to match NoteContent's schema
        )

        # ActionsTool no longer satisfies tools/base.py's Tool protocol (a
        # plain run(tool_args) -> ToolResult call). Its per-turn logic is now
        # a compiled LangGraph subgraph (see graph.py in this package),
        # registered directly as the "run_actions_tool" node in the
        # orchestrator's graph instead of being wrapped in a run() call --
        # see orchestrator/graph.py and orchestrator/tool_router.py.
        self.subgraph = build_actions_subgraph(self.llm, self.sampling_params)
