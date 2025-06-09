 
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
    """ Save motivational text for motivational shorts video, the text that meets task criteria  to a file with a timestamp.
    Args:
        text: The text to be saved. To avoid syntax errors, wrap the string in triple quotes 
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
                model_id = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Ministral-8B-Instruct-2410",
                load_in_4bit=True,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype="auto",
                max_new_tokens=15000 ,
                use_flash_attn=True,
            )



        loaded_reasoning_agent_prompts = r'C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Prompt_templates\smolagents_prompt.yaml'
        with open(loaded_reasoning_agent_prompts, 'r', encoding='utf-8') as f:
                Prompt_template = yaml.safe_load(f)

        final_answer = FinalAnswerTool()
        
        Reasoning_Text_Agent = CodeAgent(
            model=Global_model,
            tools=[SaveMotivationalText,final_answer],
            max_steps=1,
            verbosity_level=4,
            prompt_templates=Prompt_template,
            stream_outputs=True,
        )

        #Should save 1 line quote [[4607.62s - 4620.62s] the only reason you are not living the life you want is because you day by day keep feeding the life that you dont want to live.]
        chunktest_1 = """
            [00:00] Hey, how's your day going?
            [00:02] Mine was kinda slow, not gonna lie.
            [00:04] But something hit me earlier.
            [00:06] I realized that comfort is a silent killer.  
            [00:08] Every day you avoid discomfort, you also avoid growth.  
            [00:10] You don't need to feel ready—you just need to move forward anyway.  
            [00:12] That's what separates those who stay stuck and those who evolve.  
             """
        #saveable:  [00:08]
        #saveable:  [00:10] 
        #saveable:  [00:12] 
        ##MODEL SUCCESSFULLY PASSED TEST (1)

        ### Should save multi line quote [ [4607.62s - 4620.62s] the only reason you are not [3684.45s - 3696.95s] living the life you want  [3696.95s - 3707.23s] is because you day by day keep feeding  the life that you dont want to live.]
        chunk_test2 = """
                        [00:00] You know, sometimes I wonder.
                        [00:02] What if all of this is just noise?
                        [00:04] Then I remember:  
                        [00:06] The world is loud, but clarity is a choice.  
                        [00:08] You don’t need everyone to get it—just yourself.  
                        [00:10] Peace starts when you stop explaining yourself to people committed to misunderstanding you.
                        """
        #saveable:  [00:06]
        #saveable:  [00:08] 
        #saveable:  [00:10] 


        chunk_test3 = """
                        [00:00] I used to wait for motivation to start.
                        [00:02] Until I learned this:
                        [00:04] Discipline is doing it even when you're not in the mood.
                        [00:06] That's the difference between a goal and a habit.
                        [00:08] Anyone can want something. Few are willing to work when it's boring.
        """
        #saveable:  [00:04]
        #saveable:  [00:06] 
        #saveable:  [00:08] 



        chunk_test4 = """
                    [00:00] I just feel tired sometimes.
                    [00:02] Not physically, just... everything, you know?
                    [00:04] Like I’m carrying something invisible.
                    [00:06] But I show up anyway.  
                    [00:08] Because discipline isn’t about feeling good—it's about commitment.  
                    [00:10] And the hard days are where real strength is built.  
                      """
        #saveable:  [00:06]
        #saveable:  [00:08] 
        #saveable:  [00:10] 



        chunk_test5_nosaving = """
                    [00:00] I just feel tired sometimes.
                    [00:02] Not physically, just... everything, you know?
                    [00:04] Like I’m carrying something invisible.
                    [00:08] life is hard sometimes i think
                    [00:10] it feels like it  
            """
        
        #savable: nothing

        chunk_test6_nosaving = """
                    [00:00] the mindset is kinda medium
                    [00:02] remember yesterday when i say
                    [00:04] did you want a coffe?
                    [00:08] but you did not want it 
                    """

       #savable: nothing

        task = f"""
                    You are an expert at identifying  powerful, share-worthy snippets from motivational podcast transcripts.
                    Your job is to:

                    1. Read the transcript chunk below and internally reason through its overall message.
                    2. Extract only those lines or passages that:
                        • Stand alone with full context (no missing setup).  
                        • Pack a punch of advice, insight, or inspiration.  
                        • Are memorable enough to anchor a motivational short video.

                    Do NOT save generic fluff—the transcript as a whole is already motivational.

                    ––––––––––––––––––––––––––––––––––––––––––––––––––––––
                    Helper definitions (WHAT TO SAVE):

                    • Inspiring Text:  
                    – Definition: Uplifts, excites and encourages the listener , sparks hope or ambition It can be a message or story that motivates the reader to take action, achieve goals, or view things in a new way..  
                    – Example: “When you face your fears, you discover the strength you never knew you had.”

                    • Wisdom Text:  
                    – Definition: Condensed life lessons, timeless truths or a collection of teachings, stories, or sayings that offer guidance on living a good and fulfilling life, often with an emphasis on morality, virtue, and achieving happiness.  
                    – Example: “Success isn’t a destination— it’s a mindset you cultivate every day.”

                    • Motivational Text:  
                    – Definition: Calls to action that push toward growth or change a piece of writing, usually concise, that is designed to inspire, uplift, or encourage an individual to pursue goals or overcome obstacles. These texts can take various forms, including quotes, stories, speeches, and even articles or letters. The core function of motivational text is to evoke a positive mindset, instill confidence, and drive action. .  
                    – Example: “Stop waiting for the perfect moment; create it with your own two hands.”

                    • Quote Text:  
                    – Definition: Motivational quotes are concise, Short, standalone sentences, aphorisms, inspiring phrases designed to encourage and uplift individuals, often helping them stay focused, determined, and positive.  
                    – Example: “Fall seven times, stand up eight.”

                    • Personal Growth Text:  
                    – Definition: Insights into self-development, mindset shifts or Personal growth, also known as self-development, is a continuous process of improving oneself in various aspects of life, including mental, emotional, social, and physical well-being.  
                    – Example: “Your only competition is the person you were yesterday.”
                    ––––––––––––––––––––––––––––––––––––––––––––––––––––––
                    Helper definitions (WHAT TO AVOID):
                    • Avoid vague compliments or praise (e.g., “That was great!”)  
                    • Avoid cliché or overused phrases with no fresh angle  
                    • Avoid long-winded storytelling—opt for concise impact  
                    • Avoid context-less lyrics, jokes, or tangents  
                    • Avoid purely descriptive narration (e.g., “Today we talked about gratitude…”)
                    • Avoid generic motivational fluff that sounds good but adds no new insight  
                    • Avoid surface-level pep-talks lacking depth or practical advice  
                    • Avoid motivational filler that pads out time without delivering a punch  
                    • Avoid text that lacks enough context or “power lines”—snippets that sound strong but don’t stand on their own
                    ––––––––––––––––––––––––––––––––––––––––––––––––––––––
                    Helper definitions (Content to Exclude)
                    • Avoid uncertain or hedged language (e.g., “I think this might help you…”).
                       Why: You want bold statements, not wishy-washy suggestions.

                    • Avoid questions or rhetorical setups (e.g., “Have you ever felt stuck?”).
                        Why: Clips that ask questions leave viewers hanging—they need resolution or insight.

                    • Avoid internal monologue or 2nd-person reflection (e.g., “I was thinking to myself…”).
                        Why: We need universal truth or advice, not personal journaling.

                    • Avoid overly technical or niche jargon (e.g., “Using an autoencoder to reconstruct latent features…”).
                        Why: Keeps it accessible and broadly relatable.

                    • Avoid excessive qualifiers or filler words (e.g., “Basically,” “Honestly,” “You know…”).
                        Why: Cuts to the core message.

                    • Avoid monotone observations (e.g., “This is what happened next.”).
                        Why: We want emotional hooks, not neutral narration.

                    •  Avoid back-pedaling or negations (e.g., “Don’t think this is too hard.”).
                        Why: Positive, proactive language lands stronger.

                    • Avoid over-explaining the obvious (e.g., “We all know that hard work leads to success.”).
                        Why: Seeks fresh angles, not restated clichés.

                    •  Avoid multi-step instructions (e.g., “First do this, then do that…”).
                        Why: Short videos need one clear takeaway, not a how-to tutorial.

                    •  Avoid embedded jokes or humorous asides (e.g., “I almost died laughing…”).
                        Why: Humor can derail the motivational momentum unless it’s directly tied to the insight.
                    ––––––––––––––––––––––––––––––––––––––––––––––––––––––----------------------------------------

                    Important considerations:
                    • Each saved snippet must be **self-contained**: if watched alone, the viewer still “gets it.”  
                    • Prioritize **novel insights**—phrases they’ll remember and possibly share.  
                    • Ensure each snippet works as a **standalone motivational short**: concise, punchy, and immediately impactful.
                    ––––––––––––––––––––––––––––––––––––––––––––––––––––––------------------------------------------------------

                    Now analyze the chunk below and extract any qualifying snippets based on the criteria above.  
                    If there are none, reply clearly with: “No qualifying snippets found.” in the `final_answer` tool
                    ––––––––––––––––––––––––––––––––––––––––––––––––––––––------------------------------------------------------

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