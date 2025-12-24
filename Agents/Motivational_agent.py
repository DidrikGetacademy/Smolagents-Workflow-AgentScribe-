from utility.Custom_Agent_Tools import create_motivationalshort
from smolagents import  FinalAnswerTool, CodeAgent
import yaml
from utility.log import log
from utility.clean_memory import clean_get_gpu_memory
import utility.Global_state

def Motivational_analytic_agent(transcript_path,agent_txt_saving_path):
    """Agent that analyzes text  from transcript by reading it (chunk for chunk) --->  (saves Quote identified in podcast transcript."""
    from utility.Custom_Agent_Tools import ChunkLimiterTool
    from App import wait_for_proccessed_video_complete
    log(f"✅Transcript_Reasoning_AGENT (Running)")
    from App import Reload_and_change_model
    global Global_model
    Global_model =  Reload_and_change_model("gpt-5-minimal",message="Reloading model to -> gpt-5 before running [Motivational_analytic_agent]")
    loaded_reasoning_agent_prompts = r'C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Agents\Prompt_templates\Motivational Analytic Agent\System_prompt.yaml'
    with open(loaded_reasoning_agent_prompts, 'r', encoding='utf-8') as f:
            Prompt_template = yaml.safe_load(f)


    final_answer = FinalAnswerTool()

    Reasoning_Text_Agent = CodeAgent(
            model=Global_model,
            tools=[create_motivationalshort,final_answer],
            max_steps=2,
            prompt_templates=Prompt_template,
            use_structured_outputs_internally=True,
            verbosity_level=1,
            stream_outputs=False,

        )


    chunk_limiter = ChunkLimiterTool()


    reasoning_log = []
    while True:
        reasoning_log.clear()

        chunk = chunk_limiter.forward(file_path=transcript_path, max_chars=10000)

        if not chunk.strip():
                log(f"\nTranscript Path is (EMPTY)\n -{transcript_path}")
                del Reasoning_Text_Agent
                utility.Global_state.chunk_proccesed_event.set()
                clean_get_gpu_memory(threshold=0.8)
                wait_for_proccessed_video_complete(utility.Global_state.video_task_que)
                utility.Global_state.chunk_proccesed_event.clear()
                log(f"done with work. exiting transcript reasoning agent to retrieve the next items from the queue.")
                break


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
                    additional_args={"text_file": agent_txt_saving_path},
                )

        clean_get_gpu_memory(threshold=0.1)




