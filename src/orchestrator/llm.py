# Interacts first-hand with the language model
import os
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from orchestrator.prompts import (
    build_tool_decision_prompt,
    build_response_prompt,
    build_mem_update_prompt,
    build_filename_prompt,
)
from orchestrator import memory_manager
from schemas.tool_call import ToolCall
from schemas.tool_schema import TOOL_SCHEMA

class OrchestratorLLM():

    # Initialization settings currently set to that of VLLM_RAG
    def __init__(self):

        # RAG tool's vLLM engine loads on the same GPU (see tools/ragTool/config.py),
        # so this fraction must leave room for that model too instead of assuming
        # the whole device is available. gemma-4-E4B-it's own weights alone take
        # ~15.28 GiB (measured via vLLM's model-loading log), so this fraction must
        # clear that bar before any KV cache/overhead is even considered -- on a
        # 32 GiB card, 0.6 gives ~19.2 GiB (weights + ~4 GiB headroom).
        gpu_memory_utilization = float(os.environ.get("ORCHESTRATOR_GPU_MEMORY_UTILIZATION", "0.6"))

        self.llm = LLM(
            model="google/gemma-4-E4B-it",  # Gemma 4 E4B as the LLM brain of the orchestrator
            gpu_memory_utilization=gpu_memory_utilization,  # reserve this fraction of VRAM for the KV cache and runtime buffers (tweak via ORCHESTRATOR_GPU_MEMORY_UTILIZATION if memory runs out when running)
            max_model_len=4096  # sets the maximum context window that vLLM will allocate KV cache for
        )  # Use this to install gemma4:26B quantized via Huggingface

        self.decision_sampling_params = SamplingParams(
            temperature=0,  # temperture: randomness
            top_p=1.0,  # top_p; nucleus sampling
            max_tokens=512,  # Max tokens outputted
            repetition_penalty = 1.1,  # Penalty to apply if tokens continue repeating.
            structured_outputs=StructuredOutputsParams(json=TOOL_SCHEMA)  # Forces output to match ToolCall's schema
        )

        self.response_sampling_params = SamplingParams(
            temperature=0.4,  # some randomness allowed; this stage writes free text, not JSON
            top_p=0.95,  # top_p; nucleus sampling
            max_tokens=512,  # Max tokens outputted
            repetition_penalty = 1.1  # Penalty to apply if tokens continue repeating.
        )

        self.mem_update_sampling_params = SamplingParams(
            temperature=0.2,  # low randomness; this stage writes a short factual summary
            top_p=0.9,  # top_p; nucleus sampling
            max_tokens=180,  # memory notes are meant to stay short
            repetition_penalty=1.1  # Penalty to apply if tokens continue repeating.
        )

        self.filename_sampling_params = SamplingParams(
            temperature=0.0,  # deterministic; this stage outputs a short filename slug
            top_p=1.0,  # top_p; nucleus sampling
            max_tokens=24,  # a 2-5 word slug needs very few tokens
            repetition_penalty=1.1  # Penalty to apply if tokens continue repeating.
        )

    # Phase 1 Orchestrotor LLM usage: Tool decision
    def tool_decision(self, user_message) -> ToolCall:

        memory_context = memory_manager.read_system_memory()
        unformatted_prompt = build_tool_decision_prompt(user_message, memory_context)

        # Use tokenizers to format "prompt"
        tokenizer = self.llm.get_tokenizer()

        # Formatted prompt
        formatted_prompt = tokenizer.apply_chat_template(

            unformatted_prompt,
            tokenize=False,
            add_generation_prompt=True,

        )

        output = self.llm.generate([formatted_prompt], self.decision_sampling_params)

        raw_text = output[0].outputs[0].text
        print("\nRAW MODEL OUTPUT:\n", raw_text)  # Maybe can remove this in the future

        # Guided decoding guarantees schema-conformant JSON, so parsing/validation collapses into this one call
        return ToolCall.model_validate_json(raw_text)

    # Phase 2 Orchestrotor LLM usage: Response Generation
    def generate_response(self, user_message, tool_call: ToolCall, tool_result) -> str:

        # tool_result is a ToolResult (see tools/base.py); every tool always
        # returns one, so no None-handling is needed here anymore.
        memory_context = memory_manager.read_system_memory()
        unformatted_prompt = build_response_prompt(user_message, tool_call, tool_result, memory_context)

        tokenizer = self.llm.get_tokenizer()

        formatted_prompt = tokenizer.apply_chat_template(

            unformatted_prompt,
            tokenize=False,
            add_generation_prompt=True,

        )

        output = self.llm.generate([formatted_prompt], self.response_sampling_params)

        return output[0].outputs[0].text.strip()

    # Phase 3 Orchestrotor LLM usage: Conversation memory update
    def orchestrator_mem_update(self, user_message, tool_call: ToolCall, tool_result, final_response) -> None:

        try:
            question_number = memory_manager.count_existing_entries() + 1

            unformatted_prompt = build_mem_update_prompt(user_message, tool_call, tool_result, final_response)

            tokenizer = self.llm.get_tokenizer()
            formatted_prompt = tokenizer.apply_chat_template(
                unformatted_prompt,
                tokenize=False,
                add_generation_prompt=True,
            )

            output = self.llm.generate([formatted_prompt], self.mem_update_sampling_params)
            note_text = output[0].outputs[0].text.strip() or "(no memory note generated for this turn)"

            memory_manager.append_entry(question_number, note_text)

        except Exception as e:
            # A memory-write hiccup should never take down an otherwise-successful turn.
            print(f"Warning: failed to update conversation memory ({e}).")

    # Phase 4 Orchestrotor LLM usage: Chat memory filename decision (used only at "bye" + "y")
    def decide_chat_memory_filename(self) -> str:

        memory_content = memory_manager.read_system_memory(max_entries=None)

        try:
            unformatted_prompt = build_filename_prompt(memory_content)

            tokenizer = self.llm.get_tokenizer()
            formatted_prompt = tokenizer.apply_chat_template(
                unformatted_prompt,
                tokenize=False,
                add_generation_prompt=True,
            )

            output = self.llm.generate([formatted_prompt], self.filename_sampling_params)
            return output[0].outputs[0].text.strip()

        except Exception as e:
            # memory_manager's own sanitizer falls back to a timestamp-based name
            # for an empty slug, so failing here still results in a saved file.
            print(f"Warning: failed to generate a memory filename ({e}); using a default name.")
            return ""
