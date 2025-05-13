from smolagents import TransformersModel,FinalAnswerTool,SpeechToTextTool,CodeAgent
from Agents_tools import  SaveMotivationalQuote, ChunkLimiterTool,ExtractAudioFromVideo
import torch
import os
import gc
import yaml
from rich.console import Console



video_path = r""
Chunk_saving_text_file = r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing_save_time_project\saved_transcript_storage.txt"



# prompt_template = r""
# with open(prompt_template, 'r', encoding='utf-8') as f:
#         loaded_Manager_prompt_templates = yaml.safe_load(f)





def debugging_managed_agent():
    YourTask = """
                    You are being provided with a transcript chunk to analyze. 

                    IMPORTANT: Do NOT attempt to regenerate or fetch additional context — ONLY use the chunk you are given.

                    Your objective is to act like a thoughtful human editor. Carefully read and analyze the entire chunk from start to finish. Look for motivational quotes or inspirational insights that contain emotionally valuable messages.

                    If you identify such content, save it using the `SaveMotivationalQuote` tool.  
                    If no meaningful content is found, return a final message using the `final_answer()` function with this string:
                    "I'm now ready to receive a new chunk or text to analyze."

                    ---

                    ### Key Reminders:

                    - Do NOT stop after finding the first quote.  
                    A single chunk can contain multiple valid insights — review the **entire chunk thoroughly**.
                    - One line may not be meaningful on its own, but **2–4 adjacent lines** together might form a complete motivational message.

                    ---

                    ### Reasoning Rules:

                    - Use your **own internal judgment and interpretation**, just as a human would.
                    - DO NOT use loops, regex, filters, or code to identify quotes.
                    - DO NOT rely on keywords or string matching.
                    - Think like a human editor curating quotes for a motivational book or short video.

                    ---

                    ### Quote Extraction Guidelines:

                    A valid quote must:
                    - Contain a **complete idea or insight** (8 or more words unless profoundly motivational)
                    - Convey **motivational or inspirational meaning**
                    - Have **standalone value** — it should make sense and feel impactful on its own
                    - Relate to one or more of the following themes:
                    - Personal growth
                    - Mindset shift
                    - Resilience or strength
                    - Life or success advice
                    - Overcoming adversity
                    - Deep wisdom or emotional insight
                    - Self-awareness

                    ---

                    ### Exclude the Following:

                    - Incomplete fragments or sentence starters
                    - Conversational fillers ("Yes", "Okay", "Right")
                    - Unsupported superlatives or vague praise
                    - Statements with no actionable or reflective value
                    - Quotes under 8 words (unless extremely profound)

                    ---

                    ### Combining Multiple Lines:

                    If one line alone doesn’t meet the criteria but multiple **consecutive lines together** build a complete insight, you may stitch them together to form a motivational quote.

                    ✅ Example (Valid when combined):
                    [51.48s - 52.68s] In the quiet of despair, hope whispers,  
                    [52.68s - 53.88s] embrace the storm and let it mold your heart,  
                    [53.88s - 54.36s] for every trial seeds the garden of your strength.

                    ❌ Do NOT save a short line on its own unless it is clearly impactful.

                    ---

                    ### When a Quote is Found:

                    Save the quote using the following format:

                    ```py
                    SaveMotivationalQuote(text="[timestamps] full quote here", text_file=text_file)
                    print("Reason: This message expresses a full insight on [e.g., resilience, perspective, personal growth], and meets all required criteria.")
                    ```<end_code>

                    ---

                    ### If No Quote is Found:

                    Return this as your final action:

                    ```py
                    final_answer("I'm now ready to receive a new chunk or text to analyze.")
                    ```<end_code>

                    ---

                    REMEMBER:  
                    Always consider the **entire chunk**. Don’t stop early. Look for multiple motivational insights, and only return a final answer once you've completed your full review.
                    """



    Model = TransformersModel(
            model_id=r'C:\Users\didri\Desktop\LLM-models\Qwen\Qwen2.5-7B-Instruct',
            device_map="auto",
            max_new_tokens=500,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            
        )
    
    loaded_reasoning_agent_prompts = r'C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing_save_time_project\loaded_reasoning_agent_prompts.yaml'
    with open(loaded_reasoning_agent_prompts, 'r', encoding='utf-8') as f:
            Prompt_template = yaml.safe_load(f)


    Reasoning_Text_Agent = CodeAgent(
        model=Model,
        tools=[SaveMotivationalQuote, FinalAnswerTool()],
        max_steps=100,
        verbosity_level=3,
        prompt_templates=Prompt_template, 
 

    )

    chunk_limiter = ChunkLimiterTool()

    transcripts = [
        r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing_save_time_project\transcripts_wisdom\A Process for Finding Purpose： Do THIS to Build the Life You Want ｜ Jay Shetty.txt",

    ]



    for transcript_path in transcripts:
        transcript_title = os.path.basename(transcript_path)
        print(f"\nProcessing new transcript: {transcript_path}")
        chunk_limiter.reset()
        with open(Chunk_saving_text_file, "a", encoding="utf-8") as out:
            out.write(f"\n\n--- Transcript Title: {transcript_title} ---\n")

    while True:
        try:
            # Get the next chunk
            chunk = chunk_limiter.forward(file_path=transcript_path, max_chars=1000)
            
            # Check if chunk is empty — end of file
            if not chunk.strip():
                print("✅ Finished processing current transcript.")
                break

            print(f"\n🔍 Processing chunk:\n{chunk[:200]}...\n")  # Preview first 200 chars
            
            # Run agent on the chunk
            result = Reasoning_Text_Agent.run(
                input=chunk,
                task=YourTask,
                context_var={"text_file": Chunk_saving_text_file}
            )

            # Check agent's response
            if isinstance(result, dict) and "output" in result:
                output = result["output"].strip().lower()
                if ("i'm now ready to receive a new chunk or text to analyze." in output) or ("new chunk" in output):
                    print("✅ Agent signaled it's ready for the next chunk.")
                    continue  # Fetch and process next chunk
                else:
                    print("⚠️ Agent output did not clearly signal readiness for next chunk:")
                    print(result["output"])
                    break
            else:
                print("❌ Unexpected format from agent output:")
                print(result)
                break

        except Exception as e:
            print(f"❌ Error during chunking or agent run from file {transcript_path}: {e}")
            break


        

















def test_agent_Wisdom_finder():

    user_task = (
        "YOU MUST Follow these steps to achieve the user task.\n\n"
        
        "1. You have access to the variable `video_path`. First, extract the audio using the `ExtractAudioFromVideo` tool.\n\n"

        "2. Transcribe the audio using the `transcriber` tool (an instance of `SpeechToTextTool`) by providing the path returned from the `ExtractAudioFromVideo` tool.\n"
        "Note: The transcriber (`SpeechToTextTool`) will return the path to a `.txt` file containing the full transcript of the audio. Store this path in the variable `Full_transcript_path`.\n\n"

        "3. Retrieve a chunk of the transcript by calling the `chunk_limiter` tool and passing in variable `Full_transcript_path` and a `max_chars` value (e.g., 1000).\n"
        "Note: This tool returns one small portion of the transcript at a time. It does not return the full transcript. Choose a chunk size that allows for safe reasoning (e.g., 1000 characters).\n"

        "4. Pass the current chunk to the managed agent `ReasoningAgent` by calling it as:\n"
        " `ReasoningAgent.run(task=chunk, additional_args={\"text_file\: text_file})`\n"
        "This agent will reason through the chunk and extract any motivational or inspirational quotes. Refer to its description for details on behavior and tool usage.\n"
        "Wait for the response from `ReasoningAgent` before proceeding to the next chunk.\n\n"

        "5.Repeat the process: retrieve a chunk using `chunk_limiter`, send it to `ReasoningAgent`, wait for the response, and only then retrieve the next chunk.\n"
        "IMPORTANT: You must call `chunk_limiter` **one time per cycle** and wait for the reasoning agent to finish before calling it again.\n\n"
        "Repeat the following loop:\n"
            "   - Retrieve the next chunk using `chunk_limiter`.\n"
            "   - Pass the chunk to `ReasoningAgent` and wait for its response.\n"
            "   - Continue this loop until `chunk_limiter` returns an empty string.\n"
            "   - Once the full transcript is processed and `chunk_limiter` returns an empty string, provide the final answer.\n\n"

        "Remember: Stop this loop when `chunk_limiter` returns an empty string. This indicates that the full transcript has been processed.\n\n"
        "IMPORTANT:\n"
            "- You must call `chunk_limiter` **only once per cycle**, and only after `ReasoningAgent` completes processing the previous chunk.\n"
            "- DO NOT use code like `while chunk != \"\"` — this is incorrect and will break the task flow.\n"
            "- Always explicitly call `chunk_limiter` only after `ReasoningAgent` has finished."
    )



    Reasoning_Text_Agent = CodeAgent(
                name="ReasoningAgent",
                model=model,
                description=(
                "An agent that processes transcript chunks one at a time, reasoning through each to identify motivational quotes, deep insights, wisdom, or inspiring and meaningful content in the text.",
                " If such content is found, it execute the `SaveMotivationalQuote` Tool and stores the text/quote it found and the correct timestamp in the `text_file` variable that the manager agent has access to and must provide to the ReasoningAgent."
                "which invokes the SaveMotivationalQuote tool to record it. If no relevant content is detected, the chunk is ignored."
                ),
                tools=[SaveMotivationalQuote,FinalAnswerTool()],
                max_steps=30,
                prompt_templates=None

        )


    manager_agent  = CodeAgent(
                model=model,
                tools=[ChunkLimiterTool(), SpeechToTextTool(),FinalAnswerTool()], 
                managed_agents=[Reasoning_Text_Agent],
                max_steps=30,
                verbosity_level=1,
                prompt_templates=None
            )



    Response = manager_agent.run(
                task=user_task,
                additional_args = {
                    "video_path": video_path,
                    "text_file" : Chunk_saving_text_file,
                    "Full_transcript_path": None
                }       
            )
    print(Response)

    with open(r"C:\Users\didri\Desktop\Programmering\VideoEnchancer program\text_files\Agent_logging.txt", "w") as f:
        manager_agent.logger.console = Console(file=f, force_terminal=True)
        manager_agent.visualize()
    manager_agent.visualize()
    del model



import os
import subprocess
from smolagents import SpeechToTextTool
def transcribe_audio_to_txt(video_paths):
    tool = SpeechToTextTool()
    tool.setup()

    for video_path in video_paths:
        if not os.path.isfile(video_path):
            print(f"File not found: {video_path}")
            continue

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        folder = os.path.dirname(video_path)
        audio_path = os.path.join(folder, f"{base_name}.wav")
        txt_output_path = os.path.join(folder, f"{base_name}.txt")

        # Extract audio using ffmpeg
        try:
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Overwrite output if exists
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # WAV format
                audio_path
            ]
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Extracted audio to: {audio_path}")
        except subprocess.CalledProcessError:
            print(f"Failed to extract audio from {video_path}")
            continue

        # Transcribe the audio
        try:
            result_txt_path = tool.forward({"audio": audio_path})
            # Optionally rename the transcript to desired name
            if result_txt_path != txt_output_path:
                os.rename(result_txt_path, txt_output_path)
            print(f"Transcript saved to: {txt_output_path}")
        except Exception as e:
            print(f"Transcription failed for {audio_path}: {e}")


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

  
    #test_agent_Wisdom_finder()
    try:
       debugging_managed_agent()
    except Exception as e: 
        print(f"Error: {e}")

 
