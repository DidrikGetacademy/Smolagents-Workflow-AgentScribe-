 
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
def SaveMotivationalText(text: str, text_file: str) -> None:
    """ Save motivational text for motivational shorts video, the text that meets task criteria  to a file with the exact  timestamp and the connected line.
    Args:
        text: The text to be saved. must be complete thought needing no further context to understand. To avoid syntax errors, wrap the string in triple quotes 
              when calling this function, especially if the text contains commas, quotes, or line breaks.
              Example:
              text = \"[00.23s - 00.40s] This is a quote, with commas, 'apostrophes', and line breaks. Still safe."
        text_file: The path to the file where the quote will be saved, you have access to the variable, just write text_file=text_file.
    """
    with open(text_file, "a", encoding="utf-8") as f:
            f.write("===START_QUOTE===\n")
            f.write(text.strip() + "\n")
            f.write("===END_QUOTE===\n\n")
            print(f"text: {text}")
     

def test():

        Global_model = TransformersModel(
                model_id = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\deepseek-llm-7b-chat",
                load_in_4bit=True,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype="auto",
                use_flash_attn=True,
            )



        loaded_reasoning_agent_prompts = r'C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Prompt_templates\test.yaml'
        with open(loaded_reasoning_agent_prompts, 'r', encoding='utf-8') as f:
                Prompt_template = yaml.safe_load(f)

        final_answer = FinalAnswerTool()
        
        Reasoning_Text_Agent = CodeAgent(
            model=Global_model,
            tools=[SaveMotivationalText,final_answer],
            max_steps=1,
            verbosity_level=4,
            prompt_templates=Prompt_template,
            stream_outputs=True
        )

        #Should save 1 line quote [[4607.62s - 4620.62s] the only reason you are not living the life you want is because you day by day keep feeding the life that you dont want to live.]
        chunktest_1 = """
            [284.72s - 290.84s] give yourself a mindset reset. And I'm going to explain what a mindset reset is. The simple step
            [290.84s - 295.80s] the simple steps to doing it, it's all going to make sense in a couple minutes. But I want to give
            [295.80s - 302.92s] you a preview so that as we step into the process of giving yourself a reset mentally, that you have
            [302.92s - 308.76s] a baseline understanding. So here's a preview of what we're going to talk about, okay? You have a filter
            [308.76s - 315.62s] in your brain, and I'm going to teach you using science and neuroscience, how to use this filter
            [315.62s - 320.74s] that's already in your brain to your advantage. And everything that I'm going to teach you,
            [320.84s - 325.78s] you can put to practice immediately. And what I love so much about this conversation that we're
            [325.78s - 333.04s] about to have is that you may even experience an immediate change the first time you try this
            [333.04s - 338.68s] little thing I'm going to teach you. This is so cool because the moment you experience this small
            [338.68s - 344.68s] change in your mindset, it will create momentum. It creates excitement. It creates possibility.
            [344.98s - 351.36s] There's this opening of a whole new way of thinking. But before we can get there, I want to just start
            [351.36s - 357.80s] with the basics so that you feel really empowered around the topic of mindset and around reprogramming
            [357.80s - 362.72s] this filter in your brain. So let's just start with a definition. So we're all using the same
            [362.72s - 371.06s] terminology. And let's define you and I. What is a mindset? Well, your mindset is your beliefs and your
            [371.06s - 377.18s] opinions about the way that the world works. That's the definition when you look it up. However, you know
            [377.18s - 382.22s] that I prefer metaphors. Mel Robbins is dyslexic, so she likes to be able to visualize something,
            [382.40s - 388.80s] especially when we're talking about this intellectual stuff, okay? So the metaphor that I love when it
            [388.80s - 395.16s] comes to mindset and the science-y, psychological, neurological aspect of mindset and brain programming
            [395.16s - 404.91s] is I use the metaphor sunglasses. I think about your mindset like a pair of sunglasses. So stop and
            [404.91s - 411.17s] think right now about your favorite pair of sunglasses. I have these sunglasses that I have had for almost
            [411.17s - 415.83s] 15 years. I bought them because we were going on this rafting trip and I had forgot to pack my
            [415.83s - 422.23s] sunglasses. And so I bought the only cool pair of sunglasses that they had on that turnstile thing
            [422.23s - 428.33s] on the counter. They were like 15 bucks and they were these huge black bug-eyed glasses. I feel like
            [428.33s - 434.37s] Jackie O when I wear them. So think about your favorite pair of sunglasses for just a minute. Now I want you
            [434.37s - 441.60s] to think about the lens color and think about how when you put on that pair of sunglasses, that lens on
            [441.60s - 448.82s] your favorite sunglasses, it colors and filters what you see and it gives it a tint, right? I mean,
            [448.84s - 454.88s] if you put on rose-colored sunglasses, the world has a rosy bright tint to it. If you put on amber
            [454.88s - 460.84s] sunglasses, same thing. Gray, same thing. My big black bug-eyed glasses that I just love. I feel so glamorous
            [460.84s - 469.58s] in these $15 plastic things. Everything looks crazy dark. Just really blocks everything out. Your mindset
            [469.58s - 475.25s] works the same way as a pair of sunglasses. Let's go back to the written definition of your mindset.
            [475.39s - 481.31s] Your mindset is made up of your beliefs and your opinions. And just like the lens on a pair of
            [481.31s - 488.09s] sunglasses, those beliefs and opinions that you have, they create a mindset through which you filter
            [488.09s - 494.67s] the world. And I'm going to give you a couple examples. Let's say you're a pessimistic person.
            [494.67s - 504.80s] The magic you are looking for is in the work 
            [501.24s - 506.34s] you are avoiding 
            [507.21s - 513.25s] And if you're not pessimistic, let's just think about the most pessimistic person you know.
            [514.06s - 521.20s] Someone who is always negative. They could be sitting on the beach in the Bahamas with a beautiful,
            [521.20s - 528.46s] fabulous, tropical drink in their hands. Sun is shining, crystal clear ocean, and they're annoyed
            [528.46s - 534.46s] because lunch hasn't come out yet. You know that kind of person. You've sat next to them at a wedding
            [534.46s - 540.90s] where the band is awesome. The couple is so cute and happy. Family's together. And what is this person
            [540.90s - 546.22s] doing? They're bitching about something. Some relative that's sitting all the way on the other
            [546.22s - 553.04s] side of the room. All they notice is the one thing that's wrong or irritating them. They don't even
            [553.04s - 558.70s] notice all of the amazing things that are going on around them. Isn't it interesting when I describe
            [558.70s - 565.02s] this negative, pessimistic person? You know exactly who I'm talking about. And you're probably thinking,
            [565.22s - 570.18s] dear God, do not sit them next to me at the next family wedding. I do not want to hear this,
            [570.18s - 576.14s] okay? I do not like that kind of mindset or that mood. I do not want dark colored glasses skewing the
            [576.14s - 582.53s] way that I enjoy this situation right now. And here's the craziest thing about mindset. You know
            [582.53s - 588.63s] that pessimistic person you and I were just thinking about? They have no idea that they have dark glasses
            [588.63s - 596.00s] on. This is just the way they see the world. I'm going to give you another example of mindset and
            [596.00s - 600.38s] how important this is. I want you to think about someone you work with, or maybe you go to school
            [600.78s - 607.68s] person who has a can-do attitude. No matter how tight the deadline or how rude the customer is that you
            [607.68s - 613.14s] guys are waiting on, or how much other team members are slacking off, this one person with a can-do
            [613.14s - 620.90s] attitude, they always see the bright side. Or they have this unbelievable ability to just shrug off
            [620.90s - 626.18s] the rudeness of other people or the laziness of the students that are on your group project.
             """


        task = f"""
                    You are an expert at identifying  powerful, share-worthy snippets from motivational podcast transcripts.
                    Your job is to:
                         1. Read the transcript chunk below and internally reason through its overall message.
                         2. Understand the connection between the lines. and what they are saying
                         Remember do not save any text that does not provide a complete thought. the goal is that this text will be used as a motivational shorts video. before you save the text you identified,  ask yourself if you were a listener, would you understand it.
                    2. Extract only those lines or passages that:
                        • Stand alone with full context (no missing setup).  
                        • Pack a punch of advice, insight, or inspiration.  
                        • Are memorable enough to anchor a motivational short video.
                        • Are complete thoughts or sentences, that if the text you decide to save were isolated from the rest would provide a complete thought and understanding for the listener.
                        • A complete thought is that the overall meaning of the setence/text does not miss any context like exsample of lacking context is that it starts with (and, but, etc).

                    Do NOT save generic fluff—the transcript as text in chunk is already motivational.
                    the text you choose too save needs to be complete and would result in a max  10-20 seconds motivational shorts video 
                    IF you no text is identified. nothing that could be a standalone moitvational short, only provide `final_answer` tool stating that.

                    In the 'Thought: ' sequence. Explain your goal to successfully achieve the task provided and also explain that you have understood the core task and what you are going to do.
                    Remember: DO NOT repeat or rewrite the chunk in your 'Thought:' sequence.  
                    Only summarize your plan and your reasoning. Violating this rule will cause failure.


                    Here is the chunk/text you will analyze: 

                    [chunk start] 
                    {chunktest_1}  
                    [chunk end]
                    """



        agent_txt_saving_path = r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\saving_path.txt"
        Reasoning_Text_Agent.run(task=task,additional_args={"text_file": agent_txt_saving_path})


if __name__ == "__main__":
       import torch
       torch.cuda.empty_cache()
       gc.collect()
       test()