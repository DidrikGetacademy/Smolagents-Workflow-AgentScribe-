from smolagents import TransformersModel, FinalAnswerTool, SpeechToTextTool, CodeAgent, tool
from Agents_tools import ChunkLimiterTool,Chunk_line_LimiterTool
import torch
import os
import gc
import yaml
import subprocess
from smolagents import SpeechToTextTool
from ultralytics import YOLO
import numpy as np
import mediapipe as mp
from moviepy import VideoFileClip, ImageSequenceClip, TextClip, CompositeVideoClip,vfx 
import os
import threading
import time
from typing import List
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
_current_video_url: str = None
def set_current_videourl(url: str):
    global _current_video_url
    _current_video_url = url
def get_current_videourl() -> str:
    global _current_video_url
    return _current_video_url
Chunk_saving_text_file = r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Logging_and_filepaths\saved_transcript_storage.txt"
Final_saving_text_file=r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Logging_and_filepaths\final_saving_motivational.txt"



def parse_multiline_block(block_text):
    lines = [line.strip() for line in block_text.strip().splitlines() if line.strip()]
    if not lines:
        return None, None, ""
    
    start_time, _ = parse_timestamp_line(lines[0])

    _, end_time = parse_timestamp_line(lines[-1])

    
    return start_time, end_time

def parse_timestamp_line(line):
    import re
    pattern = r"\[(\d+\.?\d*)s\s*-\s*(\d+\.?\d*)s\]"
    match = re.search(pattern, line)
    if match: 
        return float(match.group(1)), float(match.group(2))
    else:
        return None, 


def parse_subtitle_text_block(text_block):
    """
    Parses multiline subtitle text block like: 
    [2315.28s - 2319.84s] you need to descend if you want to transcend
    [2319.84s - 2322.00s] you have to let yourself go down
    ..
    returns list of (text, start_time, end_time)  tuples 
    """
    import re 
    subtitles = []
    pattern = re.compile(r"\[(\d+\.\d+)s\s*-\s*(\d+\.\d+)s\]\s*(.+)")
    lines = text_block.strip().splitlines()
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            start = float(match.group(1))
            end = float(match.group(2))
            text=match.group(3)
            subtitles.append((text, start, end))
    return subtitles 
    
@tool
def SaveMotivationalQuote_CreateShort(text: str, text_file: str) -> None:
    """Appends a motivational quote, wisdom or text with timestamp to the output text file.
    Args:
        text: The quote or message to save. To avoid syntax errors, wrap the string in triple quotes 
              when calling this function, especially if the text contains commas, quotes, or line breaks.
              Example:
              text = "This is a quote,advice to be saved.
        text_file: The path to the file where the quote will be saved.
    """
    with open(text_file, "a", encoding="utf-8") as f:
        f.write("New text saved:" + text.strip() +"\n\n")
        start_time, end_time = parse_multiline_block(text)
        print(f"start_time: {start_time}, end_time: {end_time}")
    
    if start_time is None or end_time is None:
        raise ValueError(f"Start_time or end_time is None, start_time: {start_time}, end_time: {end_time}")
  

    video_url = get_current_videourl()
    print(f"Video Url to be used for video short creation from [get_current_videourl]: {video_url}")
    try:
        print("running thread now")
        thread = threading.Thread(target=run_video_short_creation_thread, args=(video_url,start_time,end_time, text))
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


        
 




def create_short_video(video_path, start_time, end_time, video_name, subtitle_text):
    subtitles = subtitle_text
    print("subtitles: ",subtitles)
    torch.cuda.set_device(0 )
    model = YOLO("yolov8x.pt") 
    face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)


    full_video = VideoFileClip(video_path)
    clip = full_video.subclipped(start_time, end_time)

    def create_subtitles(txt,duration):
        duration = end - start
        txt_clip = TextClip(
            text=txt,
            font=r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Logging_and_filepaths\Video_clips\Cardo-Bold.ttf", 
            font_size=48,
            margin=(5, 5), 
            text_align="center" ,
            vertical_align="center",
            horizontal_align="center",
            color='white',
            stroke_color="black",
            stroke_width=0.5,
            size=(1000, None),
            method="caption",
            duration=duration
        ).with_position(('center', 0.5), relative=True)
        return txt_clip
    


    

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

    processed_clip = ImageSequenceClip(processed_frames, fps=clip.fps).with_duration(clip.duration)
 




    subtitle_clips = []
    for text, start, end in subtitles:
        clip_relative_start = start - start_time
        clip_relative_end = end - start_time
        duration = clip_relative_end - clip_relative_start
        
        if duration <= 0:
            continue

        text_clip = create_subtitles(text, duration).with_start(clip_relative_start)
        subtitle_clips.append(text_clip)
        print(f"subtitle_clips: {subtitle_clips}")

    final_clip = CompositeVideoClip(
                [processed_clip.with_position('center')] + subtitle_clips,
                size=processed_clip.size
            )
                
    final_clip.audio = clip.audio
    # Import effects from their dedicated submodules
    from moviepy.video.fx.LumContrast import LumContrast 
    from moviepy.video.fx.MultiplyColor import MultiplyColor  
    from moviepy.video.fx.FadeIn import FadeIn
    from moviepy.video.fx.FadeOut import FadeOut
 
    # Apply effects like this (NO METHOD CHAINING!)
    lum_contrast_effect = LumContrast(lum=0.5, contrast=0.2)
    final_clip = lum_contrast_effect.apply(final_clip)
    final_clip = MultiplyColor(factor=0.3).apply(final_clip) 
    final_clip = FadeIn(duration=1.0).apply(final_clip)
    final_clip = FadeOut(duration=1.0).apply(final_clip)



    output_dir = "./Logging_and_filepaths/Video_clips"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{video_name}.mp4")
    final_clip.write_videofile(
        out_path,
        codec="libx264",
        audio_codec="aac",
        bitrate="2500k",
        preset="slow",
        ffmpeg_params=[
        # "-vf",
        # "hue=h=45:s=1.3,eq=contrast=0.5:brightness=0.05"
        ]
    )

    ##optionaly add logic for upscaling videos... lets see later...
    print(f"video is completed: output path : {out_path}")
    print(subtitle_clips) 
    print(subtitles) 
    del  model
    full_video.close()
    clip.close()
    face_detector.close()




count_lock = threading.Lock()
global count
count = 0
def run_video_short_creation_thread(video_url,start_time,end_time,text):
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
               subtitles = parse_subtitle_text_block(text)
               print("Subtitles: ",subtitles)
   
               print(f"Subtitle passed to the [create_short_video] --> subtitle_text_tuple: {subtitles}")
               create_short_video(video_path=text_video_path, start_time=text_video_start_time, end_time = text_video_endtime, video_name = text_video_title,subtitle_text=subtitles)
               print(f"finnished creating video")
               text_video_path = ""
               text_video_start_time = None
               text_video_endtime = None
               text_video_title = ""
               subtitles = []
            except Exception as e:
                print(f"error during [create_short_video] {str(e)}")
        except Exception as e:
          import traceback
          print("[ERROR] in run_video_short_creation_thread:")
          traceback.print_exc()


#auto upload and schedule video on social media.
def AutoUpload_AND_Schedule():
    return 


#- en idee er at man har en ekstra agent som kan gå igjennom alle lagde videoclips til slutt og ser om det går ann og lage noe montage, en shorts video som innholder motivational quotes/advices fra videoklips (resultat blir da at agenten  velger rekkefølge  på videoen som skal slå sammen til 1. video, med tanke at (det skal være motiverende og det må passe sammen)
def create_motivational_montage_agent(clips: List[str], output_path: str):
    return 







#Agent som analyserer tekst  fra transkript ved og lese (chunk for chunk) --->  (lagrer teksten basert på (task)) #eksempel her er motiverende/quote/inspirerende
def Transcript_Reasoning_AGENT(transcripts_path):
    print(f"inside [Transcript_Reasoning_AGENT], expecting path to the transcribed text file: {transcripts_path}")


    global Global_model
    loaded_reasoning_agent_prompts = r'C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Prompt_templates\loaded_reasoning_agent_prompts.yaml'
    with open(loaded_reasoning_agent_prompts, 'r', encoding='utf-8') as f:
            Prompt_template = yaml.safe_load(f)

    tools = [FinalAnswerTool(),SaveMotivationalQuote]

    Reasoning_Text_Agent = CodeAgent(
        model=Global_model,
        tools=tools,
        max_steps=3,
        verbosity_level=1,
        prompt_templates=Prompt_template, 

    )
    chunk_limiter = ChunkLimiterTool()



    print(f"transcript_path that is being proccessed inside func[Transcript_Reasoning_Agent]: {transcripts_path}")
    transcript_title = os.path.basename(transcripts_path)
    print(f"transcript title: {transcript_title}")
    print(f"\nProcessing new transcript: {transcripts_path}")
    chunk_limiter.reset()
    with open(Chunk_saving_text_file, "a", encoding="utf-8") as out:
        out.write(f"\n--- Transcript Title: {transcript_title} ---\n")

    while True:
        try:
            chunk = chunk_limiter.forward(file_path=transcripts_path, max_chars=5000)
               
        except Exception as e:
                print(f"Error during chunking from file {transcripts_path}: {e}")
                break

        if not chunk.strip():
                print("Finished processing current transcript. Now exiting func [Transcript Reasoning Agent]")
                Verify_Agent(Chunk_saving_text_file)
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
    print(f"The text file [verify agent] will reason over, is the same txt path that [transcript_reasoning_agent] saved to, text file path:{saved_text_storage}")

    loaded_reasoning_agent_prompts = r'C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Prompt_templates\Reasoning_Again.yaml'
    with open(loaded_reasoning_agent_prompts, 'r', encoding='utf-8') as f:
            Prompt_template = yaml.safe_load(f)

    global Global_model

    Reasoning_Text_Agent = CodeAgent(
        model=Global_model,
        tools=[SaveMotivationalQuote_CreateShort, FinalAnswerTool()],
        max_steps=5,
        verbosity_level=1,
        prompt_templates=Prompt_template, 
    )
     
    chunk_limiter = Chunk_line_LimiterTool()

    transcript_path = saved_text_storage
    chunk_limiter.reset()

    while True:
            try:
                chunk = chunk_limiter.forward(file_path=transcript_path, until_phrase="New text saved")
                print(f"chunk for verify agent: {chunk} from {transcript_path}")
            except Exception as e:
                print(f"Error during chunking from file {transcript_path}: {e}")
                break

            if not chunk.strip():
                print("Finished processing current transcript.")
                set_current_videourl("")
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




import threading
import queue
gpu_lock = threading.Lock()
transcript_queue = queue.Queue()
global Global_model
def log(msg):
    now = time.strftime("%H:%M:%S")
    thread = threading.current_thread().name
    print(f"[{now}][{thread}] {msg}")


def print_gpu_stats():
    nvmlInit()
    for i in range(2):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU {i}: {info.used/1024**3:.1f}GB used / {info.total/1024**3:.1f}GB total")


def gpu_worker():
    log("GPU worker started")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')
    while True:
        item = transcript_queue.get()
        if item is None:
            print("shutdown signal received exiting GPU worker")
            break
        print_gpu_stats()

        torch.cuda.empty_cache()
        gc.collect()

        video_path_url, transcript_text_path = item
        print(f"video_path: {video_path_url} & Transcript_text_path: {transcript_text_path} is being proccessed now in [GPU_WORKER]")
        set_current_videourl(video_path_url)
        print(f"Current video_url being processed is: {video_path_url}")


        print(f"Dequeued {transcript_text_path!r}, queue size now {transcript_queue.qsize()}")
        if transcript_text_path is None:
            print(f"transcript_text_path is: {transcript_text_path}")
            print("Stopped...")
            break
        with gpu_lock:
            global Global_model
            Global_model =  TransformersModel(
            model_id=r'C:\Users\didri\Desktop\LLM-models\Qwen\Qwen2.5-7B-Instruct',
            device_map="auto",
            load_in_4bit=True,
            torch_dtype=torch.float16,
            max_new_tokens=1024,
            trust_remote_code=True,
        )
            print_gpu_stats()
            print(f"Global model device: {Global_model}")
            print(f"▶️ Running Transcript_Reasoning_AGENT on {transcript_text_path}") 
            
            Transcript_Reasoning_AGENT(transcript_text_path)
        transcript_queue.task_done()

def get_device():
    if gpu_lock.acquire(blocking=False):
        gpu_lock.release()
        print("cuda available for transcription")
        return "cuda"
    else:
        print("GPU busy — falling back to CPU")
        return "cpu"




###Transkriber audio til til tekst fra video (video -->  audio ---> text)
def transcribe_single_video(video_path):
        log(f"Starting transcription for {video_path}")

        if not os.path.isfile(video_path):
          log(f"❌ File not found: {video_path}")
          return

        set_current_videourl(video_path)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        folder = os.path.dirname(video_path)
        audio_path = os.path.join(folder, f"{base_name}.wav")
        txt_output_path = os.path.join(folder, f"{base_name}.txt")

        device = get_device()
        tool = SpeechToTextTool()
        tool.device=device
        tool.setup()
    

        if os.path.isfile(audio_path) and os.path.isfile(txt_output_path):
            log(f"Transcript already exists: {txt_output_path}, audio exists: {audio_path}")
            transcript_queue.put((video_path, txt_output_path ))
            log(f"Enqueued existing transcript for GPU processing: {txt_output_path}")
            return
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
            log(f"Extracted audio → {audio_path}")
        except subprocess.CalledProcessError:
            log(f"❌ Audio extraction failed for {video_path}")
            return
        
        try:
            result_txt_path = tool.forward({"audio": audio_path,"text_path":txt_output_path})
            if result_txt_path != txt_output_path:
                os.rename(result_txt_path, txt_output_path)
            log(f"🔊 Transcription saved → {txt_output_path}")


            transcript_queue.put((video_path, txt_output_path))
            log(f"Successfully added [video_path: {video_path} & txt_output_path: {txt_output_path}] to the queue for GPU processing")

        except Exception as e:
            print(f"Transcription failed for {audio_path}: {e}")





from concurrent.futures import ThreadPoolExecutor
if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    # import torch
    # print_gpu_stats() 
    # print(torch.cuda.get_device_name(0))  # Should show "RTX 3060 Ti"
    # print(torch.cuda.get_device_name(1))  # Should show "RTX 3060 Ti"
    # print(torch.cuda.is_available())      # Should return True
    # try:
    #   video_paths = [
    #       r"c:\Users\didri\Documents\Finding Freedom From Ego & Subconscious Limiting Beliefs ｜ Peter Crone.mp4",
    #       r"c:\Users\didri\Documents\Former Monk： “Stop Missing Your Life!” Here’s the Key To Lasting Happiness ｜ Cory Muscara.mp4",
    #       r"c:\Users\didri\Documents\How to Best Guide Your Life Decisions & Path ｜ Dr. Jordan Peterson.mp4",
    #       r"c:\Users\didri\Documents\Jordan Peterson： STOP LYING TO YOURSELF! How To Turn Your Life Around In 2024!.mp4",
    #       r"c:\Users\didri\Documents\How To Break The Habit Of Being You - Dr Joe Dispenza (4K).mp4",
    #       r"c:\Users\didri\Documents\Robert Greene： A Process for Finding & Achieving Your Unique Purpose.mp4",
    #   ]
    #   gpu_thread = threading.Thread(target=gpu_worker, name="GPU-Worker")
    #   gpu_thread.start()


    #   max_threads = 2

    #   with ThreadPoolExecutor(max_workers=max_threads) as executor:
    #       executor.map(transcribe_single_video, video_paths)
       
    #   transcript_queue.join()
      

    #   transcript_queue.put(None)
    #   gpu_thread.join()

    # except Exception as e: 
    #     print(f"Error: {e}")
    set_current_videourl(r"c:\Users\didri\Documents\Former Monk： “Stop Missing Your Life!” Here’s the Key To Lasting Happiness ｜ Cory Muscara.mp4")
    text="""
[2315.28s - 2319.84s] you need to descend if you want to transcend
[2319.84s - 2322.00s] you have to let yourself go down
[2322.00s - 2324.32s] the self-energy true self-energy
[2324.32s - 2326.88s] has a gravitational pull to it
[2326.88s - 2329.68s] but we keep ourselves from going through the layers
[2329.68s - 2331.76s] that it wants to bring us back through
[2331.84s - 2335.20s] because of our ideas of how we're supposed to practice
[2335.20s - 2336.32s] how we're supposed to behave
[2336.32s - 2337.28s] how we're supposed to think
[2337.28s - 2341.20s] and it prevents us from letting ourselves be
[2341.20s - 2343.04s] in the messiness of our experience
"""
    text_file =r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Logging_and_filepaths\final_saving_motivational.txt"
    SaveMotivationalQuote_CreateShort(text, text_file)
