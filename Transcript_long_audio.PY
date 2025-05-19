from smolagents import TransformersModel,FinalAnswerTool,SpeechToTextTool,CodeAgent
from Agents_tools import  SaveMotivationalQuote, ChunkLimiterTool,ExtractAudioFromVideo
import torch
import os
import gc
import yaml
from rich.console import Console
import subprocess
from smolagents import SpeechToTextTool


Chunk_saving_text_file = r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\saved_transcript_storage.txt"
                   


# prompt_template = r"C:\Users\didri\Desktop\Programmering\VideoEnchancer program\Assets\agent_prompts\videotext_Manger_agent_prompt.yaml"
# with open(prompt_template, 'r', encoding='utf-8') as f:
#         loaded_Manager_prompt_templates = yaml.safe_load(f)


loaded_reasoning_agent_prompts = r'C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\loaded_reasoning_agent_prompts.yaml'
with open(loaded_reasoning_agent_prompts, 'r', encoding='utf-8') as f:
        Prompt_template = yaml.safe_load(f)




def debugging_managed_agent(transcripts_path):
    global Model
    transcripts = transcripts_path
    Model = TransformersModel(
            model_id=r'C:\Users\didri\Desktop\LLM-models\Qwen\Qwen2.5-7B-Instruct',
            device_map="auto",
            load_in_8bit=True,
            torch_dtype=torch.float16,
            
        )
    Reasoning_Text_Agent = CodeAgent(
        model=Model,
        tools=[SaveMotivationalQuote, FinalAnswerTool()],
        max_steps=100,
        verbosity_level=1,
        prompt_templates=Prompt_template, 
    )

    chunk_limiter = ChunkLimiterTool()





    for transcript_path in transcripts:
        transcript_title = os.path.basename(transcript_path)
        print(f"\nProcessing new transcript: {transcript_path}")
        chunk_limiter.reset()
        with open(Chunk_saving_text_file, "a", encoding="utf-8") as out:
            out.write(f"\n--- Transcript Title: {transcript_title} ---\n")

        while True:
            try:
                chunk = chunk_limiter.forward(file_path=transcript_path, max_chars=1500)
               
            except Exception as e:
                print(f"Error during chunking from file {transcript_path}: {e}")
                break

            if not chunk.strip():
                print("Finished processing current transcript.")
                break

          
            task = f"""
            You are a human-like reader analyzing & Reading the chunk and decide if it contains motivational, inspirational, wisdom-based,  or life-changing quotes or advice.
            Look specifically for quotes or advice that:
            - Inspire action or courage
            - Share deep life lessons or universal truths
            - Teach about discipline, power, respect, or success
            - Offer practical wisdom or mindset shifts that can change how someone lives
            - Are emotionally uplifting or provoke reflection
            If you find such a quote & advice, use the `SaveMotivationalQuote` tool and include the timestamp of the quote,  here is an exsample:  SaveMotivationalQuote(quote="[3567.33s - 3569.65s] - The magic you are looking for is in the work you are avoiding.")
            then procceed with the next chunk by using `final_answer` tool if no more text is worth saving in the chunk.
            you don't need or are allowed to use any other tools then `SaveMotivationalQuote`and `final_answer`
            Here is the chunk you will analyze using only reasoning like a human: \n
            [chunk start]{chunk}[chunk end]
            """


        
            result = Reasoning_Text_Agent.run(
                task=task,
                additional_args={"text_file": Chunk_saving_text_file}
            )

            print(f"Agent response: {result}\n")
            chunk_limiter.called = False 

    # Optional logging/visualization
    with open(r"C:\Users\didri\Desktop\Programmering\VideoEnchancer program\Agent_logging.txt", "w") as f:
        Reasoning_Text_Agent.logger.console = Console(file=f, force_terminal=True)
        Reasoning_Text_Agent.visualize()






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
        transcript_text_path = []

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
        text_path = os.path.join(folder,f"{base_name}.txt")
        try:
            result_txt_path = tool.forward({"audio": audio_path,"text_path":text_path})
            # Optionally rename the transcript to desired name
            if result_txt_path != txt_output_path:
                os.rename(result_txt_path, txt_output_path)
            print(f"Transcript saved to: {txt_output_path}")
            transcript_text_path.append(txt_output_path)
            debugging_managed_agent(transcript_text_path)
            
        except Exception as e:
            print(f"Transcription failed for {audio_path}: {e}")


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    try:
       video_paths = [
           r"c:\Users\didri\Documents\The Brutal Truth About Comfort & Success - Jocko Willink (4K).mp4",
           r"c:\Users\didri\Documents\The Path To Power： How To Gain Respect & Influence - Robert Greene (4K).mp4",
           r"c:\Users\didri\Documents\The Science Of Emotional Intelligence & Self-Understanding - Dr Julie Smith.mp4",
       ]
       transcribe_audio_to_txt(video_paths)
 
    except Exception as e: 
        print(f"Error: {e}")

 
