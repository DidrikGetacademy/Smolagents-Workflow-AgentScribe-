 
from smolagents import TransformersModel, FinalAnswerTool, SpeechToTextTool, CodeAgent, tool,SpeechToTextToolCPU,    InferenceClientModel
#from Custom_Agent_Tools import ChunkLimiterTool
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
def SaveMotivationalText(text: str, text_file: str) -> None:
    """Save motivational text for motivational shorts video, the text that meets task criteria
      You must include ALL timestamps for every connected line exactly as they appear from the chunk,  
    Args:
         text: The text to save. Wrap the entire block in triple quotes if it has commas, quotes, or line breaks.
              Example:
              text = \"\"\"[00.23s - 00.40s] This is line one [00.40s - 00.60s] This is line two.\"\"\"
              Make sure to keep ALL timestamps for each text line in  the entire quote identified from the chunk.
        text_file: The path to the file where the quote will be saved, you have access to the variable, just write text_file=text_file.
    """
def test():
        # Global_model = InferenceClientModel(
        #     model_id="mistralai/Mixtral-8x22B-Instruct-v0.1",
        #     api_key="",
        #     max_tokens=None,
        # )

        Global_model = TransformersModel(
                model_id = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen\Qwen2.5-Coder-3B-Instruct-MERGED",
                trust_remote_code=True,
                device_map="auto",
                torch_dtype="auto",
                max_new_tokens=5000,
            
        )


        loaded_reasoning_agent_prompts = r'C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Prompt_templates\System_prompt_TranscriptReasoning.yaml'
        with open(loaded_reasoning_agent_prompts, 'r', encoding='utf-8') as f:
                Prompt_template = yaml.safe_load(f)

        final_answer = FinalAnswerTool()

        Verify_agent_prompt = r'C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Prompt_templates\test.yaml'
        with open(loaded_reasoning_agent_prompts, 'r', encoding='utf-8') as f:
                Verify_agent_prompt = yaml.safe_load(f)

        Reasoning_Text_Agent = CodeAgent(
            model=Global_model,
            tools=[SaveMotivationalText,final_answer],
            max_steps=1,
            verbosity_level=1,
            prompt_templates=Prompt_template,
            stream_outputs=True,

        )

        


        chunktest_1 = """
            [481.31s - 488.09s] sunglasses, those beliefs and opinions that you have, they create a mindset through which you filter
            [488.09s - 494.67s] the world. And I'm going to give you a couple examples. Let's say you're a pessimistic person.
            [507.21s - 513.25s] And if you're not pessimistic, let's just think about the most pessimistic person you know.
            [514.06s - 521.20s] Someone who is always negative. They could be sitting on the beach in the Bahamas with a beautiful,
            [521.20s - 528.46s] fabulous, tropical drink in their hands. Sun is shining, crystal clear ocean, and they're annoyed
            [528.46s - 534.46s] because lunch hasn't come out yet. You know that kind of person. You've sat next to them at a wedding
            [534.46s - 540.90s] where the band is awesome. The couple is so cute and happy. Family's together. And what is this person
            [540.90s - 546.22s] doing? They're bitching about something. Some relative that's sitting all the way on the other
            [546.22s - 553.04s] side of the room. All they notice is the one thing that's wrong or irritating them. Start small.
            [553.04s - 558.70s] Dream big. Move fast. notice all of the amazing things that are going on around them. Isn't it interesting when I describe
            [558.70s - 565.02s] this negative, pessimistic person? You know exactly who I'm talking about. And you're probably thinking, You don't need all the answers—just the courage to ask better questions.
             """


        task = f"""

        Here is the chunk you will analyze:
                                [chunk start]
                                {chunktest_1}
                                [chunk end]
                    """



        agent_txt_saving_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\test\Agent_saving_path_test.txt"
        Reasoning_Text_Agent.run(task=task,additional_args={"text_file": agent_txt_saving_path})


if __name__ == "__main__":
       import torch
       torch.cuda.empty_cache()
       gc.collect()
       test()