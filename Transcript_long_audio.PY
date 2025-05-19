from smolagents import TransformersModel,FinalAnswerTool,SpeechToTextTool,CodeAgent
from Agents_tools import  SaveMotivationalQuote, ChunkLimiterTool,ExtractAudioFromVideo,Chunk_line_LimiterTool
import torch
import os
import gc
import yaml
from rich.console import Console
import subprocess
from smolagents import SpeechToTextTool




Chunk_saving_text_file = r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\saved_transcript_storage.txt"

def debugging_managed_agent(transcripts_path):
                    


    global Model
    transcripts = transcripts_path
    Model = TransformersModel(
            model_id=r'C:\Users\didri\Desktop\LLM-models\Qwen\Qwen2.5-7B-Instruct',
            device_map="auto",
            load_in_8bit=True,
            torch_dtype=torch.float16,
            
        )
    


    loaded_reasoning_agent_prompts = r'C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Prompt_templates\loaded_reasoning_agent_prompts.yaml'
    with open(loaded_reasoning_agent_prompts, 'r', encoding='utf-8') as f:
            Prompt_template = yaml.safe_load(f)

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
            Reasoning_saved_trancript_storage()
            
        except Exception as e:
            print(f"Transcription failed for {audio_path}: {e}")









def Reasoning_saved_trancript_storage():
    Final_saving_text_file=r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\final_saving_motivational.txt"
    loaded_reasoning_agent_prompts = r'C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Prompt_templates\Reasoning_Again.yaml'
    with open(loaded_reasoning_agent_prompts, 'r', encoding='utf-8') as f:
            Prompt_template = yaml.safe_load(f)


    global Model
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

    chunk_limiter = Chunk_line_LimiterTool()

    for transcript_path in Chunk_saving_text_file:
        chunk_limiter.reset()

        while True:
            try:
                chunk = chunk_limiter.forward(file_path=transcript_path, until_phrase="New text saved")
               
            except Exception as e:
                print(f"Error during chunking from file {transcript_path}: {e}")
                break

            if not chunk.strip():
                print("Finished processing current transcript.")
                break

            task = f"""
            You are a human-like reader analyzing & Reading the chunk  that is already considered motivational by another agent, but you will make sure it is, and that it is containg  sutch text to be used for a standalone motivational short video so you must decide if it contains motivational, inspirational, wisdom-based,  or life-changing quotes or advice.
            Look specifically for quotes or advice that:
            - Inspire action or courage
            - Share deep life lessons or universal truths
            - Teach about discipline, power, respect, or success
            - Offer practical wisdom or mindset shifts that can change how someone lives
            - Are emotionally uplifting or provoke reflection
            -provides full context and understanding
            If you find such a quote & advice, use the `SaveMotivationalQuote` tool and include the timestamp of the quote,  here is an exsample:  SaveMotivationalQuote(quote="[3567.33s - 3569.65s] - The magic you are looking for is in the work you are avoiding.")
            then procceed with the next chunk/text to analyze by using `final_answer` too
            you don't need or are allowed to use any other tools then `SaveMotivationalQuote` and `final_answer`
            this is how to use the tools: 
            -SaveMotivationalQuote(text="New text saved:[858.98s - 866.98s] the magic you are looking for is in the work you are avoiding") # exsample
            -final_answer("please provide me with next text to analyze")
            Important: the text you are analyzing is already a little motivation, but you must decide if this is good enough to be used in a motivational short or not.
            Here is the text/chunk you will analyze using only reasoning like a human: \n
            [chunk start]{chunk}[chunk end]
            """
        
            result = Reasoning_Text_Agent.run(
                task=task,
                additional_args={"text_file": Final_saving_text_file}
            )

            print(f"Agent response: {result}\n")
            chunk_limiter.called = False 

    # Optional logging/visualization
    with open(r"C:\Users\didri\Desktop\Programmering\VideoEnchancer program\Agent_logging.txt", "w") as f:
        Reasoning_Text_Agent.logger.console = Console(file=f, force_terminal=True)
        Reasoning_Text_Agent.visualize()




if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    try:
     video_paths = [
         "",
         "",
         "",
     ]
     transcribe_audio_to_txt(video_paths)

    except Exception as e: 
        print(f"Error: {e}")

 
