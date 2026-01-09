from Agents.utility.Agent_tools import create_motivationalshort
from smolagents import FinalAnswerTool, CodeAgent
import yaml
from utility.log import log
from utility.clean_memory import clean_get_gpu_memory
import utility.Global_state
from Agents.utility.Agent_tools import ChunkLimiterTool
agents = ["gpt-5-minimal","gpt-5-high", "gpt-5-medium"]
from smolagents import CodeAgent
from dataclasses import is_dataclass, asdict
import json


def _to_basic(obj):
    """Recursively convert complex objects (dataclasses, pydantic, custom classes)
    into basic Python types (dict/list/str/number/bool/None) so they can be
    pretty-dumped (YAML/JSON) across multiple lines.
    """
    # Primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Dataclass
    if is_dataclass(obj):
        try:
            return _to_basic(asdict(obj))
        except Exception:
            pass

    # Pydantic v2
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return _to_basic(obj.model_dump())
        except Exception:
            pass

    # Pydantic v1
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return _to_basic(obj.dict())
        except Exception:
            pass

    # Mapping
    if isinstance(obj, dict):
        return {str(k): _to_basic(v) for k, v in obj.items()}

    # Sequence / Set
    if isinstance(obj, (list, tuple, set)):
        return [_to_basic(v) for v in obj]

    # Namedtuple-like
    if hasattr(obj, "_asdict") and callable(getattr(obj, "_asdict")):
        try:
            return _to_basic(obj._asdict())
        except Exception:
            pass

    # Fallback to object __dict__
    if hasattr(obj, "__dict__"):
        try:
            return _to_basic({k: v for k, v in vars(obj).items() if not str(k).startswith("_")})
        except Exception:
            pass

    # Last resort stringification
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def _extract_thought_code_and_tokens(result_obj):
    """Extract only the agent's thought, code, and token usage.
    Searches steps' model_output_message.content for thought/code.
    Falls back to top-level model_output_message if present.
    Returns (thought:str|None, code:str|None, input_tokens:int|None, output_tokens:int|None).
    """
    data = _to_basic(result_obj)
    thought = None
    code = None
    in_tok = None
    out_tok = None

    if isinstance(data, dict):
        # Prefer overall token usage
        tu = data.get("token_usage")
        if isinstance(tu, dict):
            in_tok = tu.get("input_tokens")
            out_tok = tu.get("output_tokens")

        # Search steps in order, keep last seen thought/code
        steps = data.get("steps") or []
        for step in steps:
            if not isinstance(step, dict):
                continue
            mom = step.get("model_output_message")
            if isinstance(mom, dict):
                content = mom.get("content")
                if isinstance(content, dict):
                    if content.get("thought"):
                        thought = content.get("thought")
                    if content.get("code"):
                        code = content.get("code")
            # Step-level token usage fallback
            stu = step.get("token_usage")
            if isinstance(stu, dict):
                in_tok = in_tok if in_tok is not None else stu.get("input_tokens")
                out_tok = out_tok if out_tok is not None else stu.get("output_tokens")

        # Top-level model_output_message fallback
        if not thought or not code:
            mom = data.get("model_output_message")
            if isinstance(mom, dict):
                content = mom.get("content")
                if isinstance(content, dict):
                    thought = thought or content.get("thought")
                    code = code or content.get("code")

    return thought, code, in_tok, out_tok

if __name__ == "__main__":
    final_answer = FinalAnswerTool()
    from App import Reload_and_change_model
    loaded_reasoning_agent_prompts = r'C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Agents\Prompt_templates\Motivational Analytic Agent\System_prompt.yaml'

    with open(loaded_reasoning_agent_prompts, 'r', encoding='utf-8') as f:
        Prompt_template = yaml.safe_load(f)

    chunk_limiter = ChunkLimiterTool()
    chunk_index = 0

    while True:
        # Get the next chunk ONLY once per round, then run all agents on it
        chunk = chunk_limiter.forward(
            file_path=r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\work_queue_folder\The Best Moments of Modern Wisdom (2025)\The Best Moments of Modern Wisdom (2025).txt",
            max_chars=5000,
        )

        if not chunk.strip():
            break

        chunk_index += 1

        # Log the transcript for this chunk once
        with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\test.txt", encoding="utf-8", mode="a") as f:
            f.write(
                f"\n=============================================================================================================================================================================\n"
                f"Chunk {chunk_index} Transcript:\n{chunk}\n"
            )

        # Run each agent on the SAME chunk
        for agent in agents:
            Global_model = Reload_and_change_model(
                agent, message=f"Reloading model to -> {agent} before running [Motivational_analytic_agent]"
            )

            Reasoning_Text_Agent = CodeAgent(
                model=Global_model,
                tools=[create_motivationalshort, final_answer],
                max_steps=1,
                prompt_templates=Prompt_template,
                use_structured_outputs_internally=True,
                verbosity_level=1,
                stream_outputs=False,
            )

            task = f"""
            Extract self-contained motivational passages or return `final_answer` with a message why no text qualify as a passage from the provided chunk of a podcast transcript featuring motivational speakers or figures,

            All selection rules must be met:
            - Standalone: Forms a complete, self-sufficient thought that stands entirely on its own without needing any prior or later context from the transcript; the passage must feel whole and coherent as an isolated unit, with a clear beginning that doesn't abruptly jump in (e.g., avoiding starts that reference 'and' or 'had' that indicate prior disscuion/context) and an ending that provides natural closure without trailing off or implying more is needed. It uses universal, timeless language (no specific events or references that assume knowledge; names are allowed only for direct attribution of the advice if the core message remains understandable and motivational without knowing the person's background). For example, a qualifying passage might be a full quote or advice segment that starts with a strong statement and ends with a conclusive insight, while a non-qualifying one could be a sentence fragment that relies on the previous or later sentence for meaning and would confuse the listener if heard alone.
            - Impact: Delivers a clear, meaningful insight, actionable advice, or wisdom focused on common motivational themes (such as perseverance, discipline, growth, resilience, mindset shift, hope, ambition, courage, self-belief, positivity, overcoming obstacles/fear, reframing challenges, goal-orientation, or building internal/external change), while strongly evoking motivational emotions or contrasts (including determination, hope, inspiration, empowerment, courage, self-belief, optimism, or even elements like sadness/reflection if they serve to highlight facts, reframe perspectives, or build toward a motivational resolution or call to action). It should be inspiring/uplifting, emotional/relatable (via universal themes or short poignant stories), and aim to boost mood, inject optimism, remind of potential, or prompt action.
            - Hook: Begins with an attention-grabbing, impactful first sentence that draws in the listener immediately.
            - Length: Approximately 10–60 but not > 60 seconds when spoken; estimate using the [start-end] timestamps from the passage's first word to last (aim for 20-100 words as a rough guide if timestamps are unclear).

            Text handling:
            - Copy the qualifying text VERBATIM from the chunk—do not rephrase, summarize, omit words, or alter spacing/punctuation.
            - If a passage spans multiple timestamped lines, merge them into ONE SaveMotivationalText call, preserving exact order, all timestamps, and formatting.

            Exclusions (do not save a passage if any of the below apply):
            - Incomplete or fragmented text, e.g., starting or ending with words like 'and' etc. that imply missing prior context.
            - Generic platitudes without depth (e.g., "Just believe in yourself" alone is too short).
            - Filler, casual anecdotes, setup/transition phrases, or anything niche/company-specific.
            - Statements that feel pulled mid-sentence or require external explanation to make sense.
            - Text that would confuse or mislead if heard without context.

            Task Outcomes:
            You must achieve one of two possible outcomes for the chunk:
            1. Save a qualifying passage using the `create_motivationalshort` tool.
            2. Reject the entire chunk by only providing the `final_answer` tool, clearly stating why no qualifying passages/lines exist in the chunk.

            You must Remember: Success means either saving qualifying passages via `SaveMotivationalText` tool (if any meet all rules) or outputting 'final_answer: Explaining reason why nothing is saved and the chunk is rejected' (if none do)—no other outputs or forced extractions. Saving any passage that relies on missing context to make sense, is incomplete/fragmented, or would confuse a listener when turned into a shorts video will result in task failure.

            Here is the chunk of transcript to analyze:
            [chunk start]
            {chunk}
            [chunk end]
                """

            result = Reasoning_Text_Agent.run(
                task=task,
                return_full_result=True,
                additional_args={
                    "text_file": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\work_queue_folder\checking.txt"
                },
            )

            # Minimal, multi-line logging: only thought, code, and token usage
            thought, code, in_tok, out_tok = _extract_thought_code_and_tokens(result)
            pretty_block = {
                "agent": agent,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "thought": thought,
                "code": code,
            }
            with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\test.txt", encoding="utf-8", mode="a") as f:
                f.write("\n---\n")
                try:
                    safe_block = _to_basic(pretty_block)
                    f.write(
                        yaml.safe_dump(
                            safe_block,
                            allow_unicode=True,
                            sort_keys=False,
                            width=120,
                            default_flow_style=False,
                        )
                    )
                except Exception:
                    # Fallback to JSON pretty print if YAML fails for any edge object types
                    safe_json = json.dumps(_to_basic(pretty_block), ensure_ascii=False, indent=2)
                    f.write(safe_json + "\n")
                f.write("...\n")

            clean_get_gpu_memory(threshold=0.1)

