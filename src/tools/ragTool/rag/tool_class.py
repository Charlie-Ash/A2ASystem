# RAG operations via vLLM
from vllm import LLM, SamplingParams
import qdrant_client

from tools.gpu_utils import check_gpu_memory
from tools.ragTool.rag.dataIngestion import build_vector_index, load_vector_index
from tools.ragTool.rag.graph import build_rag_subgraph
from tools.ragTool.config import (
    COLLECTION_NAME, QDRANT_DB_PATH, LLM_MODEL, GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN
)


class RAGTool():

    def __init__(self):

        print("RAG tool initialized.")

        # Check if vector DB already exists
        try:
            db_client = qdrant_client.QdrantClient(
                path=QDRANT_DB_PATH
            )
        except Exception as e:
            raise RuntimeError(f"Failed to open Qdrant DB at {QDRANT_DB_PATH}: {e}") from e

        if not db_client.collection_exists(COLLECTION_NAME):  # Build another DB

            print("No vector database found.")
            print("Building vector database...")

            db_client.close()
            self.index, self.db_client = build_vector_index()

            print("Embedding complete.")

        else:

            print("Vector database already exist.")
            print("Loading vector database...")

            db_client.close()
            self.index, self.db_client = load_vector_index()

            print("Loading completed.")

        # LLM loading, checks if CUDA memory is sufficient
        print("Loading RAG LLM...")
        check_gpu_memory(GPU_MEMORY_UTILIZATION)

        try:
            self.llm = LLM(
                model=LLM_MODEL,
                trust_remote_code=True,  # required for Gemma 4's custom architecture code
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                max_model_len=MAX_MODEL_LEN
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load LLM '{LLM_MODEL}': {e}") from e

        # Define sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.65,  # temperture: randomness
            top_p=0.95,  # top_p; nucleus sampling
            max_tokens=512,  # Max tokens outputted
            repetition_penalty=1.1  # Penalty to apply if tokens continue repeating.
        )

        # RAGTool no longer satisfies tools/base.py's Tool protocol (a plain
        # run(tool_args) -> ToolResult call). Its per-turn logic is now a
        # compiled LangGraph subgraph (see graph.py in this package),
        # registered directly as the "run_rag_tool" node in the orchestrator's
        # graph instead of being wrapped in a run() call -- see
        # orchestrator/graph.py and orchestrator/tool_router.py.
        self.subgraph = build_rag_subgraph(self.index, self.llm, self.sampling_params)
