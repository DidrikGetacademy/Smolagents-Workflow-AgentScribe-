

from utility.log import log
from utility.clean_memory import clean_get_gpu_memory
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
from Agents.utility.Agent_tools import create_motivationalshort
openai = OpenAI(api_key=os.getenv("OPENAI_APIKEY"))

def Motivational_analytic_agent_openai(transcript_path,agent_txt_saving_path):
    """Agent that analyzes text  from transcript by reading it (chunk for chunk) --->  saves Quote identified in podcast transcript."""
    from Agents.utility.Agent_tools import ChunkLimiterTool
    log(f"✅Transcript_Reasoning_AGENT (Running)")


    chunk_limiter = ChunkLimiterTool()

    while True:
        chunk = chunk_limiter.forward(file_path=transcript_path, max_chars=6500)
        if not chunk.strip():
                log(f"\nTranscript Path is (EMPTY)\n -{transcript_path}")
                log(f"done with work. exiting transcript reasoning agent to retrieve the next items from the queue.")
                break



        System_prompt = f"""
                YOU are an intelligent, highly precise, and dedicated assistant with strong reasoning skills, specializing in analyzing motivational podcast transcripts for extracting any qualifying passages for a impactful motivational shorts video if any is detected.
                In order to do this, You must apply the Evaluation Chain-of-Thought strategy below.

                ### Evaluation Chain-of-Thought Strategy
                - Perform a deep internal chain-of-thought reasoning and deliberate step-by-step planning before decision-making
                - Maintain context-awareness.
                - Pay close attention to detail.
                - Employ critical thinking and logical reasoning.
                - Extremely precise selection skills.
                - Understand podcast transcript/chunk clearly, by carefully reading and analyzing the entire chunk.
                - Identify and extract only the most powerful, self-contained motivational passages that would work as standalone shorts video  but only if any exist.

                ### Default Decision Bias (Very Important)
                - Always begin with the assumption: "nothing in this chunk qualifies".
                - Save a passage only if it passes **all gates** (Context-Integrity, Motivational-Content, Hook, Length, Standalone) at ≥95% confidence.
                - If you are not clearly convinced that a passage would be powerful, self-contained, and non-confusing as a short video **on its own**, you MUST reject the chunk with `final_answer` tool.
                - In any situation of doubt, partial uncertainty, or borderline quality on ANY gate, you MUST reject and output `final_answer` tool rather than risk saving a weak or confusing passage.
                - When there is a conflict between “saving more” and “being precise”, you must always prioritize precision: it is better to not save than to save even one non-qualifying passage.
                - When multiple passages in the same chunk technically pass all gates, you must save only the strongest, highest‑impact passages, not every acceptable option, You are allowed to execute the `create_motivationalshort` tool multiple times per chunk if multiple distinct, high-quality passages exist.
                - Prefer fewer, exceptional passages over many merely good ones: keep only the segments that have the clearest hook, deepest insight, and cleanest standalone arc that would not confuse a listener by missing surrounding context.

                ### Shorts medium and listener-only constraint (Critical)
                - The snippet will be used as a Short-form motivational video that will be uploaded to a motivational youtube account. The listeners has access ONLY to this saved snippet—no prior or later context from the chunk.
                - Therefore, the passage must be fully self-sufficient: understandable, impactful, and conclusive on its own.


                ### Mid-line slicing policy (Word-level start/stop)
                - Line breaks in the transcript are formatting only. You may start and end within a line at word boundaries.
                - You may start at the first word, any middle word, or the last word of a line—choose the position that creates the strongest standalone opener.
                - You may end at the first word, any middle word, or the last word of a line. You may also end at the first word(s) of the next line if that is where a natural closure occurs.
                - Start at the first complete word/phrase that yields a standalone opener (drop dangling fragments/fillers like “Yeah/Well/And” at the beginning of a line by starting later in the same line).
                - End at a natural sentence/clause completion, even if this is mid-line.
                - Keep the wording 100% verbatim inside the selected span; do not rewrite or delete words inside the span.

                ### Edge cases for mid-line starts/ends (explicit)
                - Last-word opener (allowed): If the final word of a line is the start of a new, self-contained sentence/phrase, you may start exactly at that last word.
                - First-word closer (allowed): You may end at the first word(s) of the next line when a natural closure occurs there.
                - Cross-line continuity: Cross lines freely; select only the exact substring(s) that form a self-contained, well‑punctuated span passage.

                ### Standalone opener & closer constraint
                - Hook: The first meaningful word must not be a continuation marker (“And”, “But”, “So”, “Yeah”, “Well”, etc.).
                - Hook: Avoid starting with continuation markers (“And”, “But”,  “Then” “So”, “Yeah”, “Well”, etc.). Only keep them if the sentence reads as a fresh, fully self-contained statement; if in doubt, drop them by starting at the next word.
                - Opener: Do not rely on unseen context (avoid “as I said”, “like we talked about”, or bare “this/that/it/they” without introducing the referent within the span/passage, this would clearly confuse a listener).
                - Closer: End with a complete thought (e.g., sentence end or clear concluding clause), not on “and/but/because/so…”.
                - Example: If a line reads “your body. So if you want to have high self-esteem, then earn your own self-respect.”, a valid opener is “So if you want to have high self-esteem, then earn your own self-respect.” from the same line.


                ### Timestamp Handling with `create_motivationalshort` Tool:
                1. You may save partial text from within a timestamped line. Keep the original line’s [start - end] values unchanged, and place only the selected substring from that line after the timestamp.
                2. When a passage spans several lines, concatenate multiple “[start - end] <substring>” segments in the original order.
                3. Preserve original wording and punctuation for the selected substrings; do not rewrite or normalize.
                4. Example output (spanning and trimming within lines):
                    - Good mid-line start:
                    [540.84s - 547.76s] So if you want to have high self-esteem, then earn your own self-respect.
                    - Good mid-line end across two lines:
                    [548.66s - 565.60s] I had this idea, the internal golden rule. So the golden rule says treat others the way that you should be treated, you want to be treated. The internal golden rule says treat yourself like others should have treated you. And it was a riposte to maybe people that didn't grow up with
                    [565.60s - 571.22s] unconditional love in that way.
                5. BEFORE SAVING, estimate duration using intra-line position if needed:
                    - Approximate sub-line time by linear interpolation on character or word index within the line.
                    - total_duration ≈ (last_line_end_time_adjusted_for_subline_end) − (first_line_start_time_adjusted_for_subline_start)
                    - Only save if ≤ 60s. Do not output invented sub-line timestamps; use this only for an internal length check.

                ### Timestamp Integrity Rule (CRITICAL)
                - YOU must always check before saving an passage that the timestamps are correctly preserved for each line in the passage from the source chunk.
                - Never move, swap, or reorder text across lines.
                - Each “[SSSS.SSs - SSSS.SS]” you include must be an actual correct timestamp pair from the source line, unchanged.
                - The text that follows a timestamp may be a substring of that line (start/end at word boundaries).
                - When merging multiple lines, concatenate timestamp+substring segments in the same order as the source/chunk.
                - Do not invent new timestamps or alter timestamp values. Do not attach words from one line to another line’s timestamp.


                ### Task Outcomes:
                You must achieve one of two possible outcomes for each chunk:
                1. If there are one or more qualifying passages, you must save **every** qualifying passage by calling the `create_motivationalshort` tool **once per passage** (you may call this tool multiple times in a single run, one call for each distinct qualifying passage in the chunk). Extract a self-contained motivational passage and save the passages using `create_motivationalshort` tool.
                2. If no passage qualifies, you must reject the entire chunk by not executing any tool, but responding with a message clearly stating why no qualifying passages/lines exist in the chunk. Skip tool usage and provide a final answer explaining why no passages/lines qualify for a motivational shorts video.
        """

        Developer_prompt = f"""
        All selection rules must be met:
        - Standalone: Forms a complete, self-sufficient thought that stands entirely on its own without needing any prior or later context from the transcript; the passage must be read whole and coherent as an isolated unit, with a clear beginning that doesn't abruptly jump in (e.g., avoiding starts that reference 'and' or 'had' etc that indicate prior discussion/context) and an ending that provides natural closure without trailing off or implying more is needed. It uses universal, timeless language (no specific events or references that assume knowledge; names are allowed only for direct attribution of the advice if the core message remains understandable and motivational without knowing the person's background). For example, a qualifying passage might be a full quote or advice segment that starts with a strong statement and ends with a conclusive insight, while a non-qualifying one could be a sentence fragment that relies on the previous or later sentence for meaning and would confuse the listener if heard  in a short motivational video.
        - Impact: Delivers a clear, meaningful insight, quote, actionable advice, or wisdom focused on common motivational themes (such as perseverance, discipline, growth, resilience, mindset shift, hope, ambition, courage, self-belief, positivity, overcoming obstacles/fear, reframing challenges, goal-orientation, or building internal/external change), while strongly evoking motivational emotions or contrasts (including determination, hope, inspiration, empowerment, courage, self-belief, optimism, or even elements like sadness/reflection if they serve to highlight facts, reframe perspectives, or build toward a motivational resolution or call to action). It should be inspiring/uplifting, emotional/relatable (via universal themes or short poignant stories), and aim to boost mood, inject optimism, remind of potential, or prompt action.
        - Hook: Begins with an attention-grabbing, impactful first sentence that draws in the listener immediately.
        - Length: Any passage with timestamp duration below 10.0 seconds or above 60.0 seconds MUST be rejected, regardless of content quality, Do NOT justify, reinterpret, or estimate spoken duration beyond timestamps.  estimate using the [start-end] timestamps from the passage's first word to last (aim for 20-100 words as a rough guide if timestamps are unclear).

        Exclusions (do not save a passage if any of the below apply):
        - Incomplete or fragmented text, e.g., starting or ending with words like 'and' etc. that imply missing prior context.
        - Generic platitudes without depth (e.g., "Just believe in yourself" alone is too short).
        - Filler, casual anecdotes, setup/transition phrases, or anything niche/company-specific.
        - Statements that feel pulled mid-sentence or require external explanation to make sense.
        - Text that would confuse or mislead if heard without context.

        Text handling:
        - Copy the qualifying text VERBATIM from the chunk—do not rephrase, summarize, omit words, or alter spacing/punctuation.
        - If a passage spans multiple timestamped lines, merge them into ONE SaveMotivationalText call, preserving exact order, all timestamps, and formatting.

        Here is the full path to the text_file argument for create_motivationalshort tool: {agent_txt_saving_path}

        The transcript chunk provided by the user between [chunk start] and [chunk end] is the ONLY source text for extraction and quotation,
        It will be marked between [chunk start] and [chunk end] tags.

        Do not assume any context outside the provided chunk.
        Do not reference text outside these tags.

        You must Remember: Success means either saving qualifying passages via `SaveMotivationalText` tool (if any meet all rules) or outputting 'final_answer: Explaining reason why nothing is saved and the chunk is rejected' (if none do)—no other outputs or forced extractions. Saving any passage that relies on missing context to make sense, is incomplete/fragmented, or would confuse a listener when turned into a shorts video will result in task failure.

        """
        task_prompt = f"""
        Default to rejecting this chunk; call `create_motivationalshort` only if you find any powerful, fully self-contained motivational passage/passages that passes all gates with ≥99% confidence and is 10–60s long—otherwise reject and explain why:
        [chunk start]
        {chunk}
        [chunk end]
            """



        tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "create_motivationalshort",
                    "description": "Save a single self-contained motivational passage. Rules (summary): - Verbatim only: copy text exactly as in the transcript/chunk, no paraphrasing, trimming, or cleanup.- Timestamps: include every \"[SSSS.SSs - SSSS.SSs]\" segment in order. If multi-line, merge into one string preserving order and spacing.- One call per passage: do not split a coherent passage across multiple calls.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "A complete motivational passage including all required timestamps in format [SSSS.SSs - SSSS.SSs]"
                            },
                            "reasoning": {
                              "type": "string",
                              "description": "Your internal reasoning process for selecting this specific passage from the chunk, explaining how it meets all criteria, including Standalone, Impact, Hook, Length, and Timestamp Integrity that describes why this passage is suitable for a motivational shorts video."
                            },
                            "text_file": {
                                "type": "string",
                                "description": "The full path to the text file where the passage will be saved"
                            }
                        },
                        "required": ["text", "text_file", "reasoning"]
                    }
                }
            }
        ]

        response = openai.chat.completions.create(
             model="gpt-5.2",
             messages=[
                 {"role": "system","content": System_prompt},
                 {"role": "developer","content": Developer_prompt},
                 {"role": "user", "content":task_prompt}],
             tools=tools_schema,
             tool_choice="auto",
             #temperature=0,
             reasoning_effort="xhigh",
             prompt_cache_key="motivational-transcript-agent-v2",
             prompt_cache_retention="24h",
        )

        print(response)
        log(f"response: {response}")
        with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\work_queue_folder\checking.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n\n--------------------------------------------------------------------------------------\nCHUNK:\n{chunk}\n---------------------------------------------------------------------------------------\n")


        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                log(f"message made by the model: {response.choices[0].message}")
                log(f"Tool call made by the model: {tool_call}")
                if tool_call.function.name == "create_motivationalshort":
                    import json
                    args = json.loads(tool_call.function.arguments)
                    result = create_motivationalshort(
                        text=args["text"],
                        text_file=args["text_file"],
                        reasoning=args["reasoning"]
                    )


        else:
            if response.choices[0].message.content:
                with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\work_queue_folder\checking.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n--------------------------------------------------------------------------------------\n\nNo qualifying passages found in chunk. Model explanation: {response.choices[0].message.content}\n")



        clean_get_gpu_memory(threshold=0.1)




