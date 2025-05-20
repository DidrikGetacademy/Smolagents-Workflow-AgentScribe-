from smolagents import TransformersModel,FinalAnswerTool,SpeechToTextTool,CodeAgent,tool
from Agents_tools import ChunkLimiterTool,Chunk_line_LimiterTool
import torch
import os
import gc
import yaml
from rich.console import Console
import subprocess
from smolagents import SpeechToTextTool
from ultralytics import YOLO
import numpy as np
import mediapipe as mp
from moviepy import VideoFileClip, ImageSequenceClip
import os
import threading


_current_video_url: str = None
def set_current_videourl(url: str):
    global _current_video_url
    _current_video_url = url
def get_current_videourl() -> str:
    global _current_video_url
    return _current_video_url
Chunk_saving_text_file = r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\saved_transcript_storage.txt"
Final_saving_text_file=r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\final_saving_motivational.txt"

Global_model =  TransformersModel(
            model_id=r'C:\Users\didri\Desktop\LLM-models\Qwen\Qwen2.5-7B-Instruct',
            device_map="auto",
            load_in_8bit=True,
            torch_dtype=torch.float16,
        )

@tool
def SaveMotivationalQuote_CreateShort(text: str, text_file: str) -> None:
    """Appends a motivational quote, wisdom or text with timestamp to the output text file.
    Args:
        text: The quote or message to save. To avoid syntax errors, wrap the string in triple quotes 
              when calling this function, especially if the text contains commas, quotes, or line breaks.
              Example:
              text = \"\"\"This is a quote, with commas, 'apostrophes', and line breaks.\nStill safe.\"\"\"
        text_file: The path to the file where the quote will be saved.
    """
    with open(text_file, "a", encoding="utf-8") as f:
        f.write("New text saved:" + text.strip() +"\n\n")
    import re
    match = re.search(r"\[(\d+\.\d+)s\s*-\s*(\d+\.\d+)s\]", text)
    print(f"text: {text}")
    if match:
        start_time = float(match.group(1))
        end_time= float(match.group(2))
        print(f"start_time: {start_time}, end_time: {end_time}")
  

    video_url = get_current_videourl()
    print(f"Video Url to be used for video short creation from [get_current_videourl]: {video_url}")
    try:
        print("running thread now")
        thread = threading.Thread(target=run_video_short_creation_thread, args=(video_url,start_time,end_time))
        thread.start()
    except Exception as e:
        print(f"error: {str(e)}")

@tool
def SaveMotivationalQuote(text: str, text_file: str) -> None:
    """Appends a motivational quote, wisdom or text with timestamp to the output text file.
    Args:
        text: The quote or message to save. To avoid syntax errors, wrap the string in triple quotes 
              when calling this function, especially if the text contains commas, quotes, or line breaks.
              Example:
              text = \"\"\"This is a quote, with commas, 'apostrophes', and line breaks.\nStill safe.\"\"\"
        text_file: The path to the file where the quote will be saved.
    """
    with open(text_file, "a", encoding="utf-8") as f:
        f.write("New text saved:" + text.strip() +"\n\n")
        
 

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

    print(f"video is completed: output path : {out_path}")
    full_video.close()
    clip.close()
    face_detector.close()

count_lock = threading.Lock()
global count
count = 0
def run_video_short_creation_thread(video_url,start_time,end_time):
        global count
        with count_lock:
            current_count = count
            count += 1
        try:
            print(f"RUNNING --> [run_video_short_creation_thead]: video_url: {video_url}, start_time: {start_time}, end_time: {end_time}")
            count += 1
            text_video_path = video_url
            text_video_start_time = start_time
            text_video_endtime = end_time
            text_video_title = "short1" + str(current_count)
            try:
               create_short_video(video_path=text_video_path, start_time=text_video_start_time, end_time = text_video_endtime,video_name = text_video_title)
               print(f"finnished creating video")
            except Exception as e:
                print(f"error during [create_short_video]")
        except Exception as e:
          import traceback
          print("[ERROR] in run_video_short_creation_thread:")
          traceback.print_exc()






#Agent som analyserer tekst  fra transkript ved og lese (chunk for chunk) --->  (lagrer teksten basert på (task)) #eksempel her er motiverende/quote/inspirerende
def Transcript_Reasoning_AGENT(transcripts_path):
    print(f"inside [Transcript_Reasoning_AGENT], expecting path to the transcribed text file: {transcripts_path}")


    global Global_model
    loaded_reasoning_agent_prompts = r'C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Prompt_templates\loaded_reasoning_agent_prompts.yaml'
    with open(loaded_reasoning_agent_prompts, 'r', encoding='utf-8') as f:
            Prompt_template = yaml.safe_load(f)

    Reasoning_Text_Agent = CodeAgent(
        model=Global_model,
        tools=[SaveMotivationalQuote, FinalAnswerTool()],
        max_steps=100,
        verbosity_level=1,
        prompt_templates=Prompt_template, 
    )
    chunk_limiter = ChunkLimiterTool()


    for transcript_path in transcripts_path:
        print(f"transcript_path [transcript_path in transcripts]: {transcript_path}")
        transcript_title = os.path.basename(transcript_path)
        print(f"transcript title: {transcript_title}")
        print(f"\nProcessing new transcript: {transcript_path}")
        chunk_limiter.reset()
        with open(Chunk_saving_text_file, "a", encoding="utf-8") as out:
            out.write(f"\n--- Transcript Title: {transcript_title} ---\n")

        while True:
            try:
                chunk = chunk_limiter.forward(file_path=transcript_path, max_chars=2000)
               
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
            NOTE: 1 line in the chunk might not provide full context, but  multiple lines in a chunk can provide full context & valuable quote to be saved, so consider reasoning and think over the entire chunk when answering.
            If you find such a quote & advice, use the `SaveMotivationalQuote` tool and include the timestamp of the quote,  
            here is an exsample:  SaveMotivationalQuote(quote="[3567.33s - 3569.65s] - The magic you are looking for is in the work you are avoiding.",text_file=text_file)
            then procceed with the next chunk by using `final_answer` tool if no more text is worth saving in the chunk.
            you don't need or are allowed to use any other tools then `SaveMotivationalQuote`and `final_answer`
            Here is the chunk you will analyze using only reasoning like a human: \n
            [chunk start]{chunk}[chunk end]
            """
            result = Reasoning_Text_Agent.run(
                task=task,
                additional_args={"text_file": Chunk_saving_text_file}
            )
            print(f"[Path to where the [1. reasoning agent ] saves the motivational quotes  ]: {Chunk_saving_text_file}")
            print(f"Agent response: {result}\n")
            chunk_limiter.called = False 










#Agent som verifiserer tekst som er lagret, (dobbel sjekk lagret text)
def Verify_Agent(saved_text_storage):
    print(f"expects the text file [1. reasoning agent] saved the motivational quotes to, so it can verify the saved quotes: path:{saved_text_storage}")
    transcript_path = []
    transcript_path.append(saved_text_storage)
    print(f"Transcript_path (LIST): {transcript_path}")
    print(f"transcript _path (STRING): that got sent in parameter: {saved_text_storage}")
    loaded_reasoning_agent_prompts = r'C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Prompt_templates\Reasoning_Again.yaml'
    with open(loaded_reasoning_agent_prompts, 'r', encoding='utf-8') as f:
            Prompt_template = yaml.safe_load(f)

    global Global_model

    Reasoning_Text_Agent = CodeAgent(
        model=Global_model,
        tools=[SaveMotivationalQuote_CreateShort, FinalAnswerTool()],
        max_steps=100,
        verbosity_level=2,
        prompt_templates=Prompt_template, 
    )

    chunk_limiter = Chunk_line_LimiterTool()

    for transcript_path in transcript_path:
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
            If you find such a quote & advice, use the `SaveMotivationalQuote_CreateShort` tool and include the timestamp of the quote,  here is an exsample:  SaveMotivationalQuote_CreateShort(quote="[3567.33s - 3569.65s] - The magic you are looking for is in the work you are avoiding.",text_file=text_file)
            then procceed with the next chunk/text to analyze by using `final_answer` too
            you don't need or are allowed to use any other tools then `SaveMotivationalQuote_CreateShort` and `final_answer`
            this is how to use the tools: 
            -SaveMotivationalQuote_CreateShort(text="New text saved:[858.98s - 866.98s] the magic you are looking for is in the work you are avoiding",text_file=text_file) # exsample
            -final_answer("please provide me with next text to analyze")
            Important: the text you are analyzing is already a little motivation, but you must decide if this is good enough to be used in a motivational short or not.
            Here is the text/chunk you will analyze using only reasoning like a human: \n
            [chunk start]{chunk}[chunk end]
            """
            result = Reasoning_Text_Agent.run(
                task=task,
                additional_args={"text_file": Final_saving_text_file}
            )
            print(f"[path to where the [2.agent is saving the final text that will be used for motivational short]]: {Final_saving_text_file}")
            print(result)
            chunk_limiter.called = False 







###Transkriber audio til til tekst fra video (video -->  audio ---> text)
def transcribe_audio_to_txt(video_paths):
    tool = SpeechToTextTool()
    tool.setup()
    print(f"[transcribe_audio_to_txt]: {video_paths}")
  
    for video_path in video_paths:
        if not os.path.isfile(video_path):
            print(f"File not found: {video_path}")
            continue
        video_url = video_paths[0]
        print(f"video_url=video_paths ---> {video_url} ---->[set_current_videourl(video_url)]")
        set_current_videourl(video_url)

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        folder = os.path.dirname(video_path)
        audio_path = os.path.join(folder, f"{base_name}.wav")
        txt_output_path = os.path.join(folder, f"{base_name}.txt")
        transcript_text_path = []
        if os.path.isfile(audio_path) and os.path.isfile(txt_output_path):
            transcript_text_path.append(txt_output_path)
            transcript_path = txt_output_path
            print(f"[Textfile and audio already exist: now running [Transcript_Reasoning_AGENT] with input --->]: {transcript_path}")
            Transcript_Reasoning_AGENT(transcript_text_path)
            print(f"[now running [verify_agent] with input  {Chunk_saving_text_file}]")
            Verify_Agent(Chunk_saving_text_file)
            continue
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
            import time
            time.sleep(2)
            print(f"Running [Transcript_Reasoning_AGENT] passing: {transcript_text_path}")
            Transcript_Reasoning_AGENT(transcript_text_path)
            print(f"")
            print(f"Running [verify_agent] passing : {Chunk_saving_text_file}")
            Verify_Agent(Chunk_saving_text_file)
            
        except Exception as e:
            print(f"Transcription failed for {audio_path}: {e}")
        



if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    try:
      video_path = [
          r"c:\Users\didri\AppData\Local\CapCut\Videos\test.mp4",
      ]
      transcribe_audio_to_txt(video_path)

    except Exception as e: 
        print(f"Error: {e}")

 
