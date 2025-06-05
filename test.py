 
from smolagents import TransformersModel, FinalAnswerTool, SpeechToTextTool, CodeAgent, tool,SpeechToTextToolCPU
from Agents_tools import ChunkLimiterTool
import os
import gc
import yaml
import subprocess
from smolagents import SpeechToTextTool
from moviepy import VideoFileClip, ImageSequenceClip, TextClip, CompositeVideoClip,vfx,AudioFileClip,afx
import threading
import cv2
import ffmpeg
from typing import List
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import cv2
import onnxruntime as ort
from ultralytics.utils.ops import non_max_suppression
import pynvml
import torch.nn.functional as F  
import time
import numpy as np
import cv2
import torch

@tool
def SaveMotivationalQuote(text: str, text_file: str) -> None:
    """Appends a motivational quote, wisdom or text with timestamp to the output text file.
    Args:
        text: The quote or message to save. To avoid syntax errors, wrap the string in triple quotes 
              when calling this function, especially if the text contains commas, quotes, or line breaks.
              Example:
              text = \"This is a quote, with commas, 'apostrophes', and line breaks. Still safe."
        text_file: The path to the file where the quote will be saved.
    """
    with open(text_file, "a", encoding="utf-8") as f:
            f.write("===START_QUOTE===\n")
            f.write(text.strip() + "\n")
            f.write("===END_QUOTE===\n\n")
            print(f"text: {text}")
     

def test():

        Global_model = TransformersModel(
                model_id = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Mistral-7B-Instruct-v0.2",
                load_in_4bit=True,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype="auto",
                max_new_tokens=7000,
                use_flash_attn=True,
            )



        loaded_reasoning_agent_prompts = r'C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\debug_performance\test_prompt3_template.yaml'
        with open(loaded_reasoning_agent_prompts, 'r', encoding='utf-8') as f:
                Prompt_template = yaml.safe_load(f)

        final_answer = FinalAnswerTool()
        
        Reasoning_Text_Agent = CodeAgent(
            model=Global_model,
            tools=[SaveMotivationalQuote,final_answer],
            max_steps=2,
            verbosity_level=4,
            planning_interval=1,
            prompt_templates=Prompt_template,
            stream_outputs=True,
        )

        #Should save 1 line quote [[4607.62s - 4620.62s] the only reason you are not living the life you want is because you day by day keep feeding the life that you dont want to live.]
        chunktest_1 = """
                [4253.42s - 4261.19s] And every time I see a heart, it's a reminder. Oh yeah, my brain will tell me what I want it to tell
                [4261.19s - 4272.25s] Hey, it's Mel. Thank you so much for being here. If you enjoyed that video, by God, please subscribe
                [4272.25s - 4279.26s] the only reason you are not living the life you want is because you day by day keep feeding the life that you dont want to live.
                [4279.26s - 4283.96s] just horseshit. I'm so sick of it. And I'm sure you're sick of it too. And you know, I'm sharing all
                [4286.25s - 4293.42s] amazing stuff coming. Thank you so much for sending this stuff to your friends and your family.
             """

        ### Should save multi line quote [ [4607.62s - 4620.62s] the only reason you are not [3684.45s - 3696.95s] living the life you want  [3696.95s - 3707.23s] is because you day by day keep feeding  the life that you dont want to live.]
        chunk_test2 = """
                        [4459.35s - 4465.25s] I saw you in the news paper the other day.
                        [4465.25s - 4471.03s] the only reason you are not
                        [4471.03s - 4476.29s] living the life you want
                        [3696.95s - 3707.23s] is because you day by day keep feeding  the life that you dont want to live.
                        [4476.29s - 4486.22s] up in a household where you didn't feel that way, you felt safe, you felt secure, you went to an
                        [4486.22s - 4494.66s] elementary school or a middle school
                        [4494.66s - 4502.70s] yourself, I'll tell you what, I love you. Even if you don't believe in yourself, that's okay. I believe

                        """

        task = f"""Your task is to carefully read the following chunk from a motivational podcast transcript. 

                    - If you find one or more powerful quotes or messages, save them using your provided tools.
                    - If you do NOT find any powerful quotes or messages in this chunk, please respond with 'final_answer' and ask for the next chunk.

                    [chunk start]
                    {chunk_test2}
                    [chunk end]

                    Please begin by explaining your reasoning step-by-step before concluding whether any quotes or messages are present.
                    """
        agent_txt_saving_path = r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\work_queue_folder\Mindset Reset： Take Control of Your Mental Habits ｜ The Mel Robbins Podcast\agent_saving_path.txt"
        Reasoning_Text_Agent.run(task=task,additional_args={"text_file": agent_txt_saving_path})


if __name__ == "__main__":
       import torch
       torch.cuda.empty_cache()
       gc.collect()
       test()