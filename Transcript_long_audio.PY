from smolagents import TransformersModel,FinalAnswerTool,SpeechToTextTool,CodeAgent
from Agents_tools import ExtractAudioFromVideo, SaveMotivationalQuote, ChunkLimiterTool
import torch
import os
import gc
from smolagents.models import CTranslate2Model
import yaml
from rich.console import Console




video_path = r""
Chunk_saving_text_file = r"C:\Users\didri\Desktop\Programmering\VideoEnchancer program\text_files\saved_transcript_storage.txt"


prompt_template = r"C:\Users\didri\Desktop\Programmering\VideoEnchancer program\Assets\agent_prompts\videotext_Manger_agent_prompt.yaml"
with open(prompt_template, 'r', encoding='utf-8') as f:
        loaded_Manager_prompt_templates = yaml.safe_load(f)

loaded_reasoning_agent_prompts = r'C:\Users\didri\Desktop\Programmering\VideoEnchancer program\Assets\agent_prompts\loaded_reasoning_agent_prompts.yaml'
with open(loaded_reasoning_agent_prompts, 'r', encoding='utf-8') as f:
        loaded_reason_agent_prompts = yaml.safe_load(f)


def get_model(CTranslate = True):
    global model
    if CTranslate:
        model = CTranslate2Model(
            model_id=r"C:\Users\didri\Desktop\Programmering\VideoEnchancer program\AI-onnx\mistral_ct2",
            device="cpu",
            compute_type="int8_float16",

            load_in_8bit=True,  
                        )
       
    else:
        model = TransformersModel(
            model_id=r'C:\Users\didri\Desktop\Programmering\AiCognetics Nettside\AI-Agent\Phi-3-mini-128k-instruct',
            load_in_8bit=True,
            device_map="cpu",
        )

    return model




def debugging_managed_agent():
    global model

    Reasoning_Text_Agent = CodeAgent(
        name="ReasoningAgent",
        model=model,
        description=(
            "An agent that processes transcript chunks one at a time, reasoning through each to identify motivational quotes, deep insights, wisdom, or inspiring and meaningful content in the text. "
            "If such content is found, it uses the `SaveMotivationalQuote` tool to store the quote and timestamp in the `text_file` provided by the manager agent. If no relevant content is found, the chunk is ignored."
        ),
        tools=[SaveMotivationalQuote, FinalAnswerTool()],
        max_steps=15,
        prompt_templates=loaded_reason_agent_prompts, 

    )

    chunk_limiter = ChunkLimiterTool()

    transcripts = [
        r"C:\Users\didri\Desktop\Programmering\VideoEnchancer program\text_files\transcripts_wisdom\A Process for Finding Purpose： Do THIS to Build the Life You Want ｜ Jay Shetty.txt",
        r"C:\Users\didri\Desktop\Programmering\VideoEnchancer program\text_files\transcripts_wisdom\KENDALL JENNER Opens Up About Anxiety, Insecurity, & How To Be Truly Happy! ｜ Jay Shetty.txt",
        r"C:\Users\didri\Desktop\Programmering\VideoEnchancer program\text_files\transcripts_wisdom\KOBE BRYANT'S LAST GREAT INTERVIEW On How To FIND PURPOSE In LIFE ｜ Kobe Bryant & Jay Shetty.txt",
        r"C:\Users\didri\Desktop\Programmering\VideoEnchancer program\text_files\transcripts_wisdom\Lewis Howes： This ONE Mindset Shift Will Make Financial Freedom Your Reality (Not What You Think...).txt",
        r"C:\Users\didri\Desktop\Programmering\VideoEnchancer program\text_files\transcripts_wisdom\Mel Robbins： The ‘Let Them Theory’ (A Life-Changing Hack That 15M People Can’t Stop Talking About).txt",
        r"C:\Users\didri\Desktop\Programmering\VideoEnchancer program\text_files\transcripts_wisdom\The Business Expert： The SECRET FORMULA That Launches Billion-Dollar Companies!.txt",
        r"C:\Users\didri\Desktop\Programmering\VideoEnchancer program\text_files\transcripts_wisdom\The Motivation Expert： Why You're FAILING To Achieve Your Goals (& What To Do About It!).txt",
        r"C:\Users\didri\Desktop\Programmering\VideoEnchancer program\text_files\transcripts_wisdom\Will Smith's LIFE ADVICE On Manifesting Success Will CHANGE YOUR LIFE  ｜ Jay Shetty.txt"
    ]

    text_file = r'C:\Users\didri\Desktop\Programmering\VideoEnchancer program\ManagedAgent_Storage.txt'

    for transcript_path in transcripts:
        transcript_title = os.path.basename(transcript_path)
        print(f"\nProcessing new transcript: {transcript_path}")
  
        with open(text_file, "a", encoding="utf-8") as out:
            out.write(f"\n\n--- Transcript Title: {transcript_title} ---\n")

        while True:
            try:
                chunk = chunk_limiter.forward(file_path=transcript_path, max_chars=1000)
                chunk_limiter.reset()
            except Exception as e:
                print(f"Error during chunking from file {transcript_path}: {e}")
                break

            if not chunk.strip():
                print("Finished processing current transcript.")
                break

            print(f"Processing chunk:\n{chunk}...")

            YourTask = (
                "You are being provided with a transcript chunk to analyze. "
                "Do NOT attempt to fetch or regenerate the chunk — use only what you are given.\n\n"
                "Your job is to act like a thoughtful human editor. Carefully analyze the following text chunk for motivational quotes or inspiring insights. "
                "If you find a valid quote, save it using the `SaveMotivationalQuote` tool. If no meaningful content is found, print: "
                "'I'm now ready to receive a new chunk or text to analyze.' and call the `final_answer()` function with a message indicating that you're ready for the next chunk.\n\n"
                f"--- Transcript Chunk Start ---\n{chunk.strip()}\n--- Transcript Chunk End ---"
            )

            result = Reasoning_Text_Agent.run(
                task=YourTask,
                additional_args={"text_file": text_file}
            )

            print(f"Agent response: {result}\n")
            chunk_limiter.called = False  # allow re-calling forward()

    # Optional logging/visualization
    with open(r"C:\Users\didri\Desktop\Programmering\VideoEnchancer program\Agent_logging.txt", "w") as f:
        Reasoning_Text_Agent.logger.console = Console(file=f, force_terminal=True)
        Reasoning_Text_Agent.visualize()

    # Cleanup
    del Reasoning_Text_Agent
    del model
    gc.collect()
    torch.cuda.empty_cache()




# def test_agent_Wisdom_finder():
#     global model
#     user_task = (
#         "YOU MUST Follow these steps to achieve the user task.\n\n"
        
#         "1. You have access to the variable `video_path`. First, extract the audio using the `ExtractAudioFromVideo` tool.\n\n"

#         "2. Transcribe the audio using the `transcriber` tool (an instance of `SpeechToTextTool`) by providing the path returned from the `ExtractAudioFromVideo` tool.\n"
#         "Note: The transcriber (`SpeechToTextTool`) will return the path to a `.txt` file containing the full transcript of the audio. Store this path in the variable `Full_transcript_path`.\n\n"

#         "3. Retrieve a chunk of the transcript by calling the `chunk_limiter` tool and passing in variable `Full_transcript_path` and a `max_chars` value (e.g., 1000).\n"
#         "Note: This tool returns one small portion of the transcript at a time. It does not return the full transcript. Choose a chunk size that allows for safe reasoning (e.g., 1000 characters).\n"

#         "4. Pass the current chunk to the managed agent `ReasoningAgent` by calling it as:\n"
#         " `ReasoningAgent.run(task=chunk, additional_args={\"text_file\: text_file})`\n"
#         "This agent will reason through the chunk and extract any motivational or inspirational quotes. Refer to its description for details on behavior and tool usage.\n"
#         "Wait for the response from `ReasoningAgent` before proceeding to the next chunk.\n\n"

#         "5.Repeat the process: retrieve a chunk using `chunk_limiter`, send it to `ReasoningAgent`, wait for the response, and only then retrieve the next chunk.\n"
#         "IMPORTANT: You must call `chunk_limiter` **one time per cycle** and wait for the reasoning agent to finish before calling it again.\n\n"
#         "Repeat the following loop:\n"
#             "   - Retrieve the next chunk using `chunk_limiter`.\n"
#             "   - Pass the chunk to `ReasoningAgent` and wait for its response.\n"
#             "   - Continue this loop until `chunk_limiter` returns an empty string.\n"
#             "   - Once the full transcript is processed and `chunk_limiter` returns an empty string, provide the final answer.\n\n"

#         "Remember: Stop this loop when `chunk_limiter` returns an empty string. This indicates that the full transcript has been processed.\n\n"
#         "IMPORTANT:\n"
#             "- You must call `chunk_limiter` **only once per cycle**, and only after `ReasoningAgent` completes processing the previous chunk.\n"
#             "- DO NOT use code like `while chunk != \"\"` — this is incorrect and will break the task flow.\n"
#             "- Always explicitly call `chunk_limiter` only after `ReasoningAgent` has finished."
#     )



#     Reasoning_Text_Agent = CodeAgent(
#                 name="ReasoningAgent",
#                 model=model,
#                 description=(
#                 "An agent that processes transcript chunks one at a time, reasoning through each to identify motivational quotes, deep insights, wisdom, or inspiring and meaningful content in the text.",
#                 " If such content is found, it execute the `SaveMotivationalQuote` Tool and stores the text/quote it found and the correct timestamp in the `text_file` variable that the manager agent has access to and must provide to the ReasoningAgent."
#                 "which invokes the SaveMotivationalQuote tool to record it. If no relevant content is detected, the chunk is ignored."
#                 ),
#                 tools=[SaveMotivationalQuote,FinalAnswerTool()],
#                 max_steps=30,
#                 prompt_templates=loaded_managed_agent_prompts

#         )


#     manager_agent  = CodeAgent(
#                 model=model,
#                 tools=[ExtractAudioFromVideo,ChunkLimiterTool(), SpeechToTextTool(),FinalAnswerTool()], 
#                 managed_agents=[Reasoning_Text_Agent],
#                 max_steps=30,
#                 verbosity_level=1,
#                 prompt_templates=loaded_Manager_prompt_templates
#             )



#     Response = manager_agent.run(
#                 task=user_task,
#                 additional_args = {
#                     "video_path": video_path,
#                     "text_file" : Chunk_saving_text_file,
#                     "Full_transcript_path": None
#                 }       
#             )
#     print(Response)

#     with open(r"C:\Users\didri\Desktop\Programmering\VideoEnchancer program\text_files\Agent_logging.txt", "w") as f:
#         manager_agent.logger.console = Console(file=f, force_terminal=True)
#         manager_agent.visualize()
#     manager_agent.visualize()
#     del model



# import os
# import subprocess
# from smolagents import SpeechToTextTool
# def transcribe_audio_to_txt(video_paths):
#     tool = SpeechToTextTool()
#     tool.setup()

#     for video_path in video_paths:
#         if not os.path.isfile(video_path):
#             print(f"File not found: {video_path}")
#             continue

#         base_name = os.path.splitext(os.path.basename(video_path))[0]
#         folder = os.path.dirname(video_path)
#         audio_path = os.path.join(folder, f"{base_name}.wav")
#         txt_output_path = os.path.join(folder, f"{base_name}.txt")

#         # Extract audio using ffmpeg
#         try:
#             ffmpeg_cmd = [
#                 "ffmpeg",
#                 "-y",  # Overwrite output if exists
#                 "-i", video_path,
#                 "-vn",  # No video
#                 "-acodec", "pcm_s16le",  # WAV format
#                 audio_path
#             ]
#             subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             print(f"Extracted audio to: {audio_path}")
#         except subprocess.CalledProcessError:
#             print(f"Failed to extract audio from {video_path}")
#             continue

#         # Transcribe the audio
#         try:
#             result_txt_path = tool.forward({"audio": audio_path})
#             # Optionally rename the transcript to desired name
#             if result_txt_path != txt_output_path:
#                 os.rename(result_txt_path, txt_output_path)
#             print(f"Transcript saved to: {txt_output_path}")
#         except Exception as e:
#             print(f"Transcription failed for {audio_path}: {e}")


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    global model
    model = get_model()
  
    #test_agent_Wisdom_finder()
    debugging_managed_agent()
    del model


#     video_paths = [
#     r"c:\Users\didri\AppData\Local\CapCut\Videos\Will Smith's LIFE ADVICE On Manifesting Success Will CHANGE YOUR LIFE  ｜ Jay Shetty.mp4",
# ]
#     transcribe_audio_to_txt(video_paths)
 
