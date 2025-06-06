 
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
                model_id = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Ministral-8B-Instruct-2410",
                load_in_4bit=True,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype="auto",
                max_new_tokens=7000,
                use_flash_attn=True,
            )



        loaded_reasoning_agent_prompts = r'C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Prompt_templates\Prompt1.yaml'
        with open(loaded_reasoning_agent_prompts, 'r', encoding='utf-8') as f:
                Prompt_template = yaml.safe_load(f)

        final_answer = FinalAnswerTool()
        
        Reasoning_Text_Agent = CodeAgent(
            model=Global_model,
            tools=[SaveMotivationalQuote,final_answer],
            max_steps=1,
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

        task = f"""   Your task is to carefully read the following chunk from a motivational podcast transcript.
            you must read/reason carefully. because alot of the text is motivational, and very few text per chunk if any at all is worth saving. 
            DO not save any incomplete sentence, or text that does not provide a full standalone text that provides powerful messages as a standalone.. 
            When analyzing the chunk, look 3–4 lines before and after any line that feels “almost” meaningful to ensure you’re capturing a complete thought. Sometimes a message only becomes powerful when read with its full lead-up or conclusion. 
            You must read and reason like a human (chain of thought), evaluating not just for motivational tone, but for clarity, completeness, and impact. Much of the content may sound positive, but very little of it is actually worth saving.
            Do NOT save if: The sentence is incomplete, vague, or lacks full context. It starts with filler or linking words like: "And that's true...", "Because every day...", "Which means...", "That’s why..." The message cannot stand alone and make sense or impact without previous lines.
           

             Identify valid text to save:
                Quotes:  Short, impactful sentences or phrases often attributed to famous people, authors, or anonymous sources that inspire, motivate, or provoke powerful thought. They stand alone as a complete sentence and are easily memorable, making them perfect for sharing as brief motivational messages or social media posts.
                Example:
                  "Success is not final, failure is not fatal: It is the courage to continue that counts." – Winston Churchill
                  "Dream big. Start small. Act now."

                Advices: Practical recommendations or actionable tips designed to guide behavior or decision-making. These are typically prescriptive and offer clear steps or mindsets to adopt. Unlike quotes, advice usually has a direct, instructive tone and aims to help someone take positive action or avoid common pitfalls.
                    Example:
                        "When facing doubt, list three reasons why you can succeed before giving up."
                        "To build resilience, embrace challenges as learning opportunities rather than setbacks."


                Wisdom: Deeper, often philosophical insights or truths about life, human nature, or success that carry timeless value. Wisdom transcends simple advice by offering perspective or understanding gained from experience or reflection. These messages can be thought-provoking and meaningful for long-term mindset shifts.
                    Example:
                        "True strength lies not in never falling, but in rising every time we fall."
                        "Patience is the companion of wisdom."


                Personal growth: 
                    Statements or reflections focused on self-improvement, emotional development, and the journey of becoming a better version of oneself. These often encourage introspection, goal-setting, and mindset transformation. They may combine motivational and practical elements but are specifically centered on internal change and progress.
                    Example:
                        "Every day is a new chance to rewrite your story and grow beyond your limits."
                        "Personal growth begins at the end of your comfort zone."


            - If you find one or more powerful quotes or messages, analyze the chunk completly before you later save them using your provided tools in the 'Code:' sequence.
            `SaveMotivationalQuote(text="...", text_file=text_file)`
            `final_answer("...")`
            - If you do NOT find any powerful quotes or messages in this chunk, please respond with 'final_answer' and ask for the next chunk.
            you must not forget to write <end_code> before you are done writing the code


           

            Here is the chunk you will analyze using only reasoning like a human (Chain of Thought):

                    [chunk start]
                    {chunktest_1}
                    [chunk end]

             NOW please begin by analyzing and reasoning over the entire chunk and identify any potensial text worth saving by reasoning in  'planning' by using chain of thought . you must finnish with the entre chunk before you later save after <end_plan> 
             """
        agent_txt_saving_path = r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\work_queue_folder\Mindset Reset： Take Control of Your Mental Habits ｜ The Mel Robbins Podcast\agent_saving_path.txt"
        Reasoning_Text_Agent.run(task=task,additional_args={"text_file": agent_txt_saving_path})


if __name__ == "__main__":
       import torch
       torch.cuda.empty_cache()
       gc.collect()
       test()