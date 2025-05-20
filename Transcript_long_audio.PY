from smolagents import TransformersModel,FinalAnswerTool,SpeechToTextTool,CodeAgent
from Agents_tools import  SaveMotivationalQuote, ChunkLimiterTool,ExtractAudioFromVideo,Chunk_line_LimiterTool
import torch
import os
import gc
import yaml
from rich.console import Console
import subprocess
from smolagents import SpeechToTextTool
from moviepy import VideoFileClip, ImageSequenceClip
from ultralytics import YOLO
import threading


Chunk_saving_text_file = r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\saved_transcript_storage.txt"
def debugging_managed_agent(transcripts_path):
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






# video_task_queue = queue.Queue()

# def video_worker():
#     while True:
#         video_url, start_time, end_time = video_task_queue.get()  # blocks until task available
#         video_name = os.path.basename(video_url)
#         video_title = f"{video_name} - {start_time} - {end_time}"

#         # Call your video creation function (blocking)
#         create_short_video(
#             video_path=video_url,
#             start_time=start_time,
#             end_time=end_time,
#             video_name=video_title,
#         )
#         video_task_queue.task_done()

# # Start the worker thread once
# worker_thread = threading.Thread(target=video_worker, daemon=True)
# worker_thread.start()





def Reasoning_saved_trancript_storage(video_url):
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
            if "SaveMotivationalQuote" in str(result):
                  import re
                  match = re.search(r"\[(\d+\.\d+)s\s*-\s*(\d+\.\d+)s\]", result)
                  if match:
                     start_time = float(match.group(1))
                     end_time= float(match.group(2))
                  thread = threading.Thread(target=run_video_short_creation_thread, args=(video_url,start_time,end_time))## can add this logic inside savemotivationalquote func or i can run it from here, think about this tomorrow
                  thread.start()
                  start_time = None,
                  end_time=None,
                  result=None
                #video_task_queue.put((video_url, start_time, end_time))
    

            chunk_limiter.called = False 

    # Optional logging/visualization
    with open(r"C:\Users\didri\Desktop\Programmering\VideoEnchancer program\Agent_logging.txt", "w") as f:
        Reasoning_Text_Agent.logger.console = Console(file=f, force_terminal=True)
        Reasoning_Text_Agent.visualize()

def run_video_short_creation_thread(video_url,start_time,end_time):
        text_video_path = video_url
        text_video_start_time = start_time
        text_video_endtime = end_time
        text_video_title = video_url.name + " - " + str(start_time) + " - " + str(end_time)
        create_short_video(video_path=text_video_path, start_time=text_video_start_time, end_time = text_video_endtime,video_name = text_video_title)


from ultralytics import YOLO
import numpy as np
import mediapipe as mp
from moviepy import VideoFileClip, ImageSequenceClip
import os

def create_short_video(video_path, start_time, end_time, video_name):
    model = YOLO("yolov8x.pt") 
    face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)


    full_video = VideoFileClip(video_path)
    clip = full_video.subclipped(start_time, end_time)


    TARGET_W, TARGET_H = 1080, 1920
    alpha = 0.2  

  
    prev_cx, prev_cy = None, None

    def detect_and_crop_frame(frame):
        nonlocal prev_cx, prev_cy

        img = np.ascontiguousarray(frame)

     
        results = model(img, imgsz=640)[0]
        person_boxes = [
            b.xyxy.cpu().numpy().astype(int)[0]
            for b, cls in zip(results.boxes, results.boxes.cls)
            if int(cls) == 0  
        ]


        results_face = face_detector.process(frame)
        face_boxes = []
        if results_face.detections:
            for detection in results_face.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                face_boxes.append((x, y, x + w, y + h))

        h, w, _ = frame.shape

 
        person_box = None
        if person_boxes:
            person_areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in person_boxes]
            max_person_idx = np.argmax(person_areas)
            person_box = person_boxes[max_person_idx]

     
        face_box = None
        if face_boxes:
            face_areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in face_boxes]
            max_face_idx = np.argmax(face_areas)
            face_box = face_boxes[max_face_idx]

        if face_box:
            fx1, fy1, fx2, fy2 = face_box
            cx, cy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
        elif person_box:
            x1, y1, x2, y2 = person_box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        else:
            cx, cy = w // 2, h // 2

    
        if prev_cx is None or prev_cy is None:
            sx, sy = cx, cy
        else:
            sx = int(alpha * cx + (1 - alpha) * prev_cx)
            sy = int(alpha * cy + (1 - alpha) * prev_cy)
        prev_cx, prev_cy = sx, sy

       
        x0 = max(0, min(sx - TARGET_W // 2, w - TARGET_W))
        y0 = max(0, min(sy - TARGET_H // 2, h - TARGET_H))

        return frame[y0:y0+TARGET_H, x0:x0+TARGET_W]

 
    frames = list(clip.iter_frames())
    processed_frames = [detect_and_crop_frame(f) for f in frames]

    processed_clip = ImageSequenceClip(processed_frames, fps=clip.fps)
    processed_clip = processed_clip.with_audio(clip.audio)


    output_dir = "./Video_clips"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{video_name}.mp4")
    processed_clip.write_videofile(
        out_path,
        codec="libx264",
        audio_codec="aac",
        bitrate="2500k",
        preset="slow"
    )


    full_video.close()
    clip.close()
    face_detector.close()




def transcribe_audio_to_txt(video_paths):
    tool = SpeechToTextTool()
    tool.setup()

    for video_path in video_paths:
        if not os.path.isfile(video_path):
            print(f"File not found: {video_path}")
            continue
        video_url = video_paths[0]


        base_name = os.path.splitext(os.path.basename(video_path))[0]
        folder = os.path.dirname(video_path)
        audio_path = os.path.join(folder, f"{base_name}.wav")
        txt_output_path = os.path.join(folder, f"{base_name}.txt")
        transcript_text_path = []

   
        try:
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  
                "-i", video_path,
                "-vn", 
                "-acodec", "pcm_s16le", 
                audio_path
            ]
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Extracted audio to: {audio_path}")
        except subprocess.CalledProcessError:
            print(f"Failed to extract audio from {video_path}")
            continue

        text_path = os.path.join(folder,f"{base_name}.txt")
        try:
            result_txt_path = tool.forward({"audio": audio_path,"text_path":text_path})
            if result_txt_path != txt_output_path:
                os.rename(result_txt_path, txt_output_path)
            print(f"Transcript saved to: {txt_output_path}")
            transcript_text_path.append(txt_output_path)
            debugging_managed_agent(transcript_text_path)
            Reasoning_saved_trancript_storage(video_url)
            
        except Exception as e:
            print(f"Transcription failed for {audio_path}: {e}")
        







if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    try:
     video_path = [
         r"c:\Users\didri\Documents\Original\The Path To Power： How To Gain Respect & Influence - Robert Greene (4K).mp4",
     ]
     transcribe_audio_to_txt(video_path)


    except Exception as e: 
        print(f"Error: {e}")




if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    try:
     video_path = [
         r"",
         r"",
         r"",
     ]
     transcribe_audio_to_txt(video_path)


    except Exception as e: 
        print(f"Error: {e}")

 
