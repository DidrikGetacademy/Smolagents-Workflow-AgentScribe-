import sys
import os
REPO_ROOT = os.path.abspath('.')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import json
import datetime

from Agents.utility.Agent_tools import SpeechToTextToolCUDA, SpeechToText_short_creation_thread,SpeechToText_montage_creation_thread
from utility.Videocreation_pipeline.VideoCreator import create_short_video
from utility.log import log
import torchvision.transforms.functional as F
sys.modules['torchvision.transforms.functional_tensor'] = F
import torch.nn.functional as F
import subprocess
import ffmpeg
from concurrent.futures import ThreadPoolExecutor
import pynvml
import time
from utility.clean_memory import clean_get_gpu_memory
import torch
import threading
import queue
import torch
from utility.Videocreation_pipeline.blender import enhance_frames_bpy
from pydub import AudioSegment
from proglog import ProgressBarLogger
import utility.Global_state as Global_state
from queue import Queue
from utility.Videocreation_pipeline.Utility import compose_montage_clips
import threading
from utility.reload_model import Reload_and_change_model
import re
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

#---------------------------------------#
# Cleans all the log files Automatically
#---------------------------------------#
def Clean_log_onRun():
    with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\logs\log.txt", "w", encoding="UTF-8") as w:
            w.write("")


#----------------------------------#
# Full file Path's
#----------------------------------#
Chunk_saving_text_file = r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Logging_and_filepaths\saved_transcript_storage.txt"
Final_saving_text_file=r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Logging_and_filepaths\final_saving_motivational.txt"
model_path_SwinIR_color_denoise15_pth = r"c:\Users\didri\Desktop\LLM-models\Video-upscale-models\SwinIR-M_noise15.pth"
model_path_SwinIR_color_denoise15_onnx = r"c:\Users\didri\Desktop\LLM-models\Video-upscale-models\SwinIR-M_noise15.onnx"
model_path_Swin_BSRGAN_X4_pth = r"c:\Users\didri\Desktop\LLM-models\Video-upscale-models\BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
model_path_Swin_BSRGAN_X4_onnx = r"c:\Users\didri\Desktop\LLM-models\Video-upscale-models\BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.onnx"
model_path_realesgran_x2_pth = r"c:\Users\didri\Desktop\LLM-models\Video-upscale-models\RealESRGAN_x2plus.pth"




count = 0
Video_count = 0
Global_model = None
gpu_thread_offline = False
Upload_YT_count = 0



def  clear_queue(q: Queue):
     with q.mutex:
          q.queue.clear()
          q.all_tasks_done.notify_all()
          q.unfinished_tasks = 0


class MyProgressLogger(ProgressBarLogger):
    def callback(self, **changes):
        for param, value in changes.items():
            log(f"{param}: {value}")
            log(f"{param}: {value}")




def truncate_audio(audio_path, start_time, end_time, output_path, padding=4.0):
            """
            Truncate an audio file from start_time to end_time and save it to output_path.
            Adds padding seconds before and after to improve transcription accuracy.

            start_time, end_time: in seconds (float or int)
            padding: seconds to add before start_time and after end_time (default 2.0)
            """

            if not os.path.isfile(audio_path):
               log(f"[truncate_audio] ERROR: Audio file does not exist: {audio_path}")
               raise ValueError(f"Audio file does not exist: {audio_path}")

            output_dir = os.path.dirname(output_path)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            if not os.access(output_dir, os.W_OK):
                log(f"[truncate_audio] ERROR: Output directory is not writable: {output_dir}")
                raise ValueError(f"Output directory is not writable: {output_dir}")

            try:
                audio = AudioSegment.from_file(audio_path)
                audio_duration = audio.duration_seconds
                audio_size = os.path.getsize(audio_path) / 1024
                frame_rate = audio.frame_rate
                log(f"[truncate_audio] Audio metadata: duration={audio_duration:.2f}s, size={audio_size:.2f}KB, frame_rate={frame_rate}Hz")

                if start_time < 0 or end_time <= start_time or end_time > audio_duration:
                        log(f"[truncate_audio] ERROR: Invalid time range: start_time={start_time}, end_time={end_time}, audio_duration={audio_duration}")
                        raise ValueError(f"Invalid time range: start_time={start_time}, end_time={end_time}, audio_duration={audio_duration}")

                padded_start_time = max(0, start_time - padding)
                padded_end_time = min(audio_duration, end_time + padding)
                log(f"[truncate_audio] Original range: {start_time:.2f}s - {end_time:.2f}s, Padded range: {padded_start_time:.2f}s - {padded_end_time:.2f}s (padding={padding}s)")

                start_sample = round(padded_start_time * frame_rate )
                end_sample = round(padded_end_time * frame_rate )
                log(f"[truncate_audio] Calculated: start_sample={start_sample}, end_sample={end_sample}")
                log(f"start_ms: {start_sample}, end_ms: {end_sample}")


                truncated = audio.get_sample_slice(start_sample, end_sample)
                if len(truncated) == 0:
                        log(f"[truncate_audio] ERROR: Truncated audio is empty")
                        raise ValueError("Truncated audio is empty")



                truncated.export(output_path, format="wav")
                Global_state.set_current_truncated_audio_path(output_path)
                log(f"Global state: - set_current_truncated_audio_path: {output_path}")
                if not os.path.isfile(output_path):
                    log(f"[truncate_audio] ERROR: Failed to create output file: {output_path}")
                    raise ValueError(f"Failed to create output file: {output_path}")

                output_audio = AudioSegment.from_file(output_path)
                log(f"[truncate_audio] Output created: duration={output_audio.duration_seconds:.2f}s, size={os.path.getsize(output_path)/1024:.2f}KB")
                return output_path
            except Exception as e:
                 log(f"error during trunacation: {str(e)}")


#-----------------------------------------------------#
# Verifies subtitles & video start time/video end time
#-----------------------------------------------------#
def run_video_short_creation_thread(video_url,audio_path,start_time,end_time,subtitle_text,Video_Title_name=None,Video_output_path=None):
        global Video_count
        Video_count += 1
        current_count = Video_count

        try:
            log(f"RUNNING --> [run_video_short_creation_thead]: video_url: {video_url}, start_time: {start_time}, end_time: {end_time}")
            text_video_path = video_url
            video_start_time = start_time
            video_end_time = end_time


            device = "cuda"
            tool = SpeechToText_short_creation_thread(device=device)
            tool.setup()


            probe = ffmpeg.probe(audio_path)
            audio_duration = float(probe['format']['duration'])
            log(f"[run_video_short_creation_thread] Audio duration: {audio_duration}s")

            if video_start_time >= audio_duration or video_end_time > audio_duration:
                 log(f"[run_video_short_creation_thread] ERROR: Time range exceeds audio duration: start_time={video_start_time}, end_time={video_end_time}")
                 raise ValueError("Invalid time range for audio")

            audio_dir = os.path.dirname(audio_path)
            truncated_audio_path = os.path.join(audio_dir, f"truncated_{current_count}.wav")
            log(f"Truncated audio path: {truncated_audio_path}")
            try:
                audio_for_clip = truncate_audio(audio_path, video_start_time, video_end_time,truncated_audio_path)
            except Exception as e:
                log(f"Error during trunacation of audio: {e}")

            subtitle_text = re.sub(r"\[\d+\.\d+s\s*-\s*\d+\.\d+s\]", "", subtitle_text)
            subtitle_text = re.sub(r"\s+", " ",subtitle_text).strip()
            log(f"subtitletext cleaned: {subtitle_text}")


            padding_offset = 4.0
            adjusted_start = video_start_time - padding_offset
            log(f"[run_video_short_creation_thread] Adjusting timestamps for padding: original_start={video_start_time}, adjusted_start={adjusted_start}")

            result = tool.forward({"audio": audio_for_clip,"subtitle_text": subtitle_text, "original_start_time": adjusted_start, "original_end_time:": video_end_time})

            crafted_Subtitle_text = result["matched_words"]
            new_video_start_time = float(result["video_start_time"])
            new_video_end_time = float(result["video_end_time"])
            log(f"[run_video_short_creation_thread] creating video now... \n start_time: {new_video_start_time} \n end_time: {new_video_end_time},  \n subtitle_text: {crafted_Subtitle_text}")

            if Video_Title_name is None:
                Video_title = "short1" + str(current_count)
            else:
                 Video_title = Video_Title_name

            create_short_video(video_path=text_video_path, audio_path=audio_for_clip, start_time=new_video_start_time, end_time = new_video_end_time, video_name = Video_title, subtitle_text=crafted_Subtitle_text,Video_output_path=Video_output_path,order=None, YT_channel=None, middle_order=None,Montage_Flag=False)
        except Exception as e:
                log(f"[ERROR] during execution: {str(e)}")






#---------------------------------------------------------------------------#
# Retrieves Video Creation tasks from a Queue & runs creation of that video
#---------------------------------------------------------------------------#
def video_creation_worker():
    global Global_model
    Global_state.chunk_proccesed_event.wait()
    log("[video_creation_worker] Started and ready to process video tasks")
    Invalid_creation_count = 0
    Successful_video_creation_count = 0
    while True:
        task_id = None
        subtitle_text = "N/A"
        final_start_time = "N/A"
        final_end_time = "N/A"

        try:
            Success = False
            clean_get_gpu_memory(threshold=0.8)

            item, task_id = Global_state.video_task_que.get(timeout=1.0)
            if item is None:
                log(f"[video_creation_worker] Received sentinel, exiting.")
                Global_state.video_task_que.task_done(task_id)
                break

            video_url, audio_path, final_start_time, final_end_time, subtitle_text = item
            log(f"current work being proccessed/Retrieved WORK: {video_url}\naudio_path:{audio_path}\n {final_start_time}\n {final_end_time}\n {subtitle_text}\n")
        except queue.Empty:
            log(f"[video_creation_worker] Queue empty, waiting for new tasks...")
            global gpu_thread_offline
            if gpu_thread_offline:
                break
            Global_state.chunk_proccesed_event.wait()
            continue

        try:
            run_video_short_creation_thread(video_url, audio_path, final_start_time, final_end_time, subtitle_text)
            Success = True
            clean_get_gpu_memory(threshold=0.3)
        except Exception as e:
            log(f"Error during [run_video_short_creation_thread]: {str(e)}")
            continue
        finally:
            if Success:
                log(f"Video Successfully Done! Information: SubitleText: {subtitle_text} \n start_time: {final_start_time}, end_time: {final_end_time}\n Current number of Successfull video creations: [{Successful_video_creation_count}]")
                Successful_video_creation_count += 1
            else:
                Invalid_creation_count += 1
                log(f"Video Failed during Creation!\n Information: SubitleText: {subtitle_text} \n start_time: {final_start_time}, end_time: {final_end_time}\n  Current number of failed video creations: [{Invalid_creation_count}]")

            if task_id is not None:
                Global_state.video_task_que.task_done(task_id)


#-------------------------------------------------------------------------------------------#
# Listens & waits for a queue to be empty before procceeding with Transcript Reasoning Agent
#-------------------------------------------------------------------------------------------#
def wait_for_proccessed_video_complete(queue: Queue, check_interval=60):
    """
    Blocks until the queue is empty, checking every `check_interval` seconds.
    Listens & waits for a queue to be empty before procceeding with Transcript Reasoning Agent
    """
    log(f"\n\n\n\n\n\n[wait_for_proccessed_video_complete]")
    while not queue.empty():
          log(f"[wait_for_proccessed_video_complete]  waiting for video_task_que to be empty: items remaining: {queue.qsize()}")
          time.sleep(check_interval)
    log("[wait_for_proccessed_video_complete]✅ video_task_que is now empty!!!")




def Montage_short_worker():
    montage_count = 0
    jobs = {}
    jobs_lock = threading.Lock()
    orders_required = ("start", "middle", "ending")
    while True:
           try:
               item = Global_state.Montage_clip_task_Que.get()
               if item is None:
                   log("[Montage_short_worker] Received sentinel. Exiting.")
                   Global_state.Montage_clip_task_Que.task_done()
                   break

               video_path, audio_path, start_time, end_time, subtitle_text, order, montage_id, YT_channel, middle_order = item

               try:
                    audio_dir = os.path.dirname(audio_path)
                    truncated_audio_path = os.path.join(audio_dir, f"temp_truncated_montage_{montage_id}_{order}.wav")
                    _truncate_audio = truncate_audio(audio_path, start_time, end_time, output_path=truncated_audio_path)
                    tool = SpeechToText_montage_creation_thread()

                    subtitle_text = re.sub(r"\[\d+\.\d+s\s*-\s*\d+\.\d+s\]", "", subtitle_text)
                    subtitle_text = re.sub(r"\s+", " ",subtitle_text).strip()
                    tool.setup()
                    padding_offset = 2.0
                    adjusted_start = max(0, start_time - padding_offset)
                    result = tool.forward({"audio": _truncate_audio, "subtitle_text": subtitle_text, "original_start_time": adjusted_start, "original_end_time:": end_time})
                    crafted_Subtitle_text = result["matched_words"]
                    new_video_start_time = float(result["video_start_time"])
                    new_video_end_time = float(result["video_end_time"])
                    log(f"[Montage_short_worker] Processed subtitles for montage_id={montage_id}, order={order}: {crafted_Subtitle_text}")
               except Exception as e:
                    log(f"[Montage_short_worker] Error during subtitle processing: {str(e)}")


               print(f"[Montage_short_worker] Retrieved item: video_path={video_path}, audio_path={audio_path}, start_time={start_time}, end_time={end_time}, order={order}, montage_id={montage_id}, YT_channel={YT_channel}")
               order = str(order).lower().strip()
               if order not in orders_required:
                   log(f"[Montage_short_worker] Invalid order: {order}. Skipping item: {item}")
                   Global_state.Montage_clip_task_Que.task_done()
                   continue

               # Derive montage group key N from ID like N.1/N.2/N.3
               group_key = str(montage_id).split(".")[0]
               parts_dir = os.path.join(".", "Video_clips", "Montage_clips", group_key)
               os.makedirs(parts_dir, exist_ok=True)

               # Deterministic output path for the part
               # Name parts deterministically; include middle_order to avoid overwrites for multiple middles
               if order == "middle" and middle_order is not None:
                   part_basename = f"{group_key}_{order}{int(middle_order)}.mp4"
               else:
                   part_basename = f"{group_key}_{order}.mp4"
               part_output_path = os.path.join(parts_dir, part_basename)
               log(f"[Montage_short_worker] Rendering part group={group_key} order={order} -> {part_output_path}")

               # Render the individual part (this writes part_output_path, then creates RIFE file and deletes original)
               create_short_video(
                   video_path=video_path,
                   start_time=new_video_start_time,
                   end_time=new_video_end_time,
                   subtitle_text=crafted_Subtitle_text,
                   video_name=os.path.splitext(part_basename)[0],
                   Video_output_path=part_output_path,
                   YT_channel=YT_channel,
                   order=order,
                   middle_order=middle_order,
                   Montage_Flag=True

               )

               # After RIFE, the final file is "<stem>_rife.mp4"
               rife_output_path = os.path.splitext(part_output_path)[0] + "_rife.mp4"
               final_part_path = rife_output_path if os.path.exists(rife_output_path) else part_output_path
               log(f"[Montage_short_worker] Part ready: {final_part_path}")

               # Update jobs aggregator (store paths + channel per group)
               ready_to_compose = False
               group_channel = None
               with jobs_lock:
                   # Initialize aggregator for group with support for multiple middles
                   group = jobs.setdefault(
                       group_key,
                       {
                           "paths": {"start": None, "middles": {}, "ending": None},
                           "yt": None,
                       },
                   )
                   # Lock channel to the first item; warn if mismatch later
                   if group["yt"] is None:
                       group["yt"] = YT_channel
                   elif group["yt"] != YT_channel:
                       log(f"[Montage_short_worker][WARN] YT_channel mismatch within group {group_key}: stored={group['yt']} incoming={YT_channel}. Using stored.")

                   # Store path per order; for "middle" keep by middle_order
                   ord_lower = order
                   if ord_lower == "start":
                       group["paths"]["start"] = final_part_path
                   elif ord_lower == "ending":
                       group["paths"]["ending"] = final_part_path
                   else:  # middle
                       mo = int(middle_order) if middle_order is not None else 1
                       group["paths"]["middles"][mo] = final_part_path

                   # Determine if ready: need start, ending, and contiguous middle_order sequence starting at 1
                   start_ready = group["paths"]["start"] is not None
                   end_ready = group["paths"]["ending"] is not None
                   middles_dict = group["paths"]["middles"]
                   middles_ready = len(middles_dict) > 0
                   contiguous = False
                   if middles_ready:
                       keys_sorted = sorted(middles_dict.keys())
                       contiguous = keys_sorted == list(range(1, keys_sorted[-1] + 1))

                   if start_ready and end_ready and middles_ready and contiguous:
                       ready_to_compose = True
                       group_channel = group["yt"]

               # Compose when all three parts are ready
               if ready_to_compose:
                   try:
                       montage_count += 1
                       # Build ordered list: start, all middles by middle_order asc, ending
                       paths_info = jobs[group_key]["paths"]
                       middle_keys = sorted(paths_info["middles"].keys())
                       ordered_paths = [paths_info["start"]] + [paths_info["middles"][k] for k in middle_keys] + [paths_info["ending"]]
                       final_output_dir = os.path.join(".", "Video_clips", "Montage_clips")
                       os.makedirs(final_output_dir, exist_ok=True)
                       final_output_path = os.path.join(final_output_dir, f"montage_{group_key}.mp4")
                       log(f"[Montage_short_worker] Composing montage group={group_key} -> {final_output_path}")
                       output_path = compose_montage_clips(ordered_paths, final_output_path,YT_channel,_truncate_audio,video_path,start_time,end_time)
                       from utility.upload_Socialmedia import upload_MontageClip
                       log(f"[Montage_short_worker] Uploading montage to YT channel: {group_channel}")
                       model = Reload_and_change_model("gpt-5-minimal",message="reloaded model inside Montage_short_worker before upload_MontageClip")
                       upload_MontageClip(model=model, file_path=output_path, subtitle_text="Montage video", YT_channel=group_channel)

                       log(f"[Montage_short_worker] Montage composed: {final_output_path}")
                   except Exception as e:
                       log(f"[Montage_short_worker] ERROR composing montage {group_key}: {str(e)}")
                   finally:
                       with jobs_lock:
                           jobs.pop(group_key, None)

           except Exception as e:
               log(f"[Montage_short_worker] ERROR processing item: {str(e)}")
           finally:
               try:
                   Global_state.Montage_clip_task_Que.task_done()
               except Exception:
                   pass



def add_video_source_to_completed(video_name: str, clips_created: int = 0) -> str:
    """
    Adds a new video source to the completed videos database.
    This function registers a video as used by adding it to Videosources_completed.json with metadata.

    Args:
        video_name (str): The exact name of the video to add (e.g., "New Motivational Speech.mp4")
        clips_created (int): Optional number of clips created from this video (default: 0)

    Returns:
        str: A confirmation message indicating success or failure.
    """
    json_path = r"c:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\work_queue_folder\Videosources_completed.json"

    try:
        video_name = video_name.strip()

        # Load existing data or create empty dict
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}

        # Check if video already exists
        if video_name in data:
            return f"Video '{video_name}' already exists in database. Use date: {data[video_name].get('used_date', 'unknown')}"

        # Add new video with metadata
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        data[video_name] = {
            "used_date": current_date,
            "clips_created": clips_created
        }

        # Write atomically using temp file
        temp_path = json_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        os.replace(temp_path, json_path)

        log(f"Added video source to completed: {video_name}")
        return f"Successfully added video '{video_name}' to database on {current_date}."

    except json.JSONDecodeError as e:
        return f"Error parsing database file: {str(e)}"
    except Exception as e:
        return f"Error adding video source: {str(e)}"



def save_full_io_to_file(modelname: str, input_chunk: str, reasoning_steps: list[str], model_response: str, file_path: str) -> None:
    """Writes Information/logging outputs from Agent Runs to a text file."""
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"Model running: {modelname}")
        f.write("===INPUT CHUNK START===\n")
        f.write(input_chunk.strip() + "\n")
        f.write("===INPUT CHUNK END===\n\n")

        f.write("===REASONING STEPS===\n")
        for step in reasoning_steps:
            f.write(step + "\n")
        f.write("\n")

        f.write("===MODEL RESPONSE START===\n")
        f.write(model_response.strip() + "\n")
        f.write("===MODEL RESPONSE END===\n")
        f.write("------------------------------------------------------------------------\n\n\n")




#------------------------------------------------------------------------------------------#
# Extracts Audio from video path, Transcribes it. And adds information to transcript Queue
#------------------------------------------------------------------------------------------#
def transcribe_single_video(video_path, device):
    log("transcribe_single_video")

    if not os.path.isfile(video_path):
        log(f"❌ File not found: {video_path}")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))


    parent_folder = os.path.join(script_dir, "work_queue_folder")
    os.makedirs(parent_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    folder = os.path.join(parent_folder, base_name)
    os.makedirs(folder, exist_ok=True)

    txt_output_path = os.path.join(folder, f"{base_name}.txt")
    audio_path = os.path.join(folder, f"{base_name}.wav")
    agent_txt = "agent_saving_path"
    agent_text_saving_path = os.path.join(folder, f"{agent_txt}.txt")
    os.makedirs(os.path.dirname(agent_text_saving_path), exist_ok=True)

    if os.path.isfile(audio_path) and os.path.isfile(txt_output_path):
        log(f"Transcript already exists: \ntranscript_path:{txt_output_path}\n,audio exists: {audio_path}\n")
        Global_state.transcript_queue.put((video_path, txt_output_path,agent_text_saving_path,audio_path))
        log(f"Enqueued existing transcript for GPU processing:\n Video path:{video_path}\ntxt output:{txt_output_path}\nagent saving path:{agent_text_saving_path}\n audio path: {audio_path}\n")
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
        log(f"Global audio_path: {audio_path}")
        log(f"Extracted audio → {audio_path}")
    except subprocess.CalledProcessError:
        log(f"❌ Audio extraction failed for {video_path}")
        return

    try:
        start_time = time.time()
        if device == "cuda":
                got_gpu = Global_state.gpu_lock.acquire(blocking=False)
                if not got_gpu:
                    log(f"GPU busy, falling back to CPU for {video_path}")
                    device = "cpu"
                else:
                    log(f"Acquired GPU lock for {video_path}")
        else:
             got_gpu = False



        tool = SpeechToTextToolCUDA()
        tool.device = device
        tool.setup()

        try:
             log(f"starting TRANSCRIPTION")
             result_txt_path = tool.forward({"audio": audio_path, "text_path": txt_output_path, "video_path": video_path})
        except Exception as e:
             log(f"error during transcribing: {str(e)}")
        elapsed_time = time.time() - start_time
        log(f"⏱️ Transcription took {elapsed_time:.2f} seconds for {video_path} on device {device}")

        if result_txt_path != txt_output_path:
            os.rename(result_txt_path, txt_output_path)


        import shutil
        copy_text_path = os.path.join(folder,f"{base_name}_Transcriptcopy.txt")
        shutil.copyfile(txt_output_path, copy_text_path)

        log(f"🔊 Transcription saved → {txt_output_path}")

        Global_state.transcript_queue.put((video_path, txt_output_path, agent_text_saving_path,audio_path))
        Global_state.gpu_lock.release()
        del tool
        log(f"Released GPU lock after transcribing {video_path}\n")
        log(f"added video_path: {video_path}, transcript: {txt_output_path}, agent_saving_path: {agent_text_saving_path} to queue  for GPU processing")

    except Exception as e:
        log(f"Transcription failed for {audio_path}: {e}")



#------------------------------------------------------------------------------------------#
# Retrieves Work/items from the Transcript queue & runs the transcript reasoning Agent
#------------------------------------------------------------------------------------------#
def gpu_worker():
    log("GPU-Thread started (Running)\n")
    torch.backends.cudnn.benchmark = True

    log(f"Loaded Global_model on device")
    while True:
        item = Global_state.transcript_queue.get()
        if item is None:

            log("[Shutdown signal received] - exiting GPU worker")
            Global_state.transcript_queue.task_done()
            Global_state.chunk_proccesed_event.set() # signaliserer at video_creation_worker kan starte og kjøre og ikke henge på wait lengre.
            clean_get_gpu_memory(threshold=0.8)
            wait_for_proccessed_video_complete(Global_state.video_task_que) # stopper thread her og venter på at alle videoer er ferdig prossesert
            Global_state.chunk_proccesed_event.clear()
            break

        video_path_url, transcript_text_path,agent_txt_saving_path,_audio_path = item


        Global_state.set_current_audio_path(_audio_path)
        Global_state.set_current_videourl(video_path_url)
        Global_state.set_current_textfile(agent_txt_saving_path)
        log(f"Item Retrieved from (Transcript Queue) ready for Processing: Global state:\n - set_current_videourl:{video_path_url}\n, Global state: set_current_audio_path:\n - {_audio_path}\n Transcript text path - {transcript_text_path}\n agent is saving text from transcript to Global state:\n set_current_textfile:{agent_txt_saving_path}")

        try:
            from Agents.Motivational_agent import Motivational_analytic_agent
            Motivational_analytic_agent(transcript_text_path, agent_txt_saving_path)
            log(f"Transcript_Reasoning_AGENT  (EXITED)")
        finally:
            log(f"Transcript Queue (TASK DONE)\n - {transcript_text_path}")
            Global_state.transcript_queue.task_done()






def load_downloaded_video_paths() -> list[str]:
        """Load downloaded video files from work_queue_folder/videos_downloaded."""
        base_dir = os.path.join(os.path.dirname(__file__), "work_queue_folder", "videos_downloaded")
        if not os.path.isdir(base_dir):
            log(f"No downloaded videos folder found at {base_dir}")
            return []

        video_files = []
        for name in os.listdir(base_dir):
            if name.lower().endswith((".mp4", ".mkv", ".mov")):
                log(f"Found downloaded video file: {name}")
                video_files.append(os.path.join(base_dir, name))

        if not video_files:
            log("No downloaded video files found in videos_downloaded folder.")

        return video_files




if __name__ == "__main__":

    video_paths = load_downloaded_video_paths()
    if not video_paths:
        log("No videos available to process; exiting main thread.")
        sys.exit(0)

    clean_get_gpu_memory(threshold=0.1)


    worker_thread = threading.Thread(target=video_creation_worker,name="Video_creation(THREAD)")
    worker_thread.start()


    log(f"Video_paths: {len(video_paths)}")


    devices = ["cuda"]
    video_device_pairs = [(video_paths[i], devices[i % len(devices)]) for i in range(len(video_paths))]
    max_threads = 1
    start_time = time.time()


    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        start_time = time.time()
        for video_path, device in video_device_pairs:
            futures.append(executor.submit(transcribe_single_video, video_path, device))

        for future in futures:
            future.result()
            end_time = time.time()
        total_time = end_time - start_time
        log(f"start_time of threadpool: {start_time} & endtime of threadpool = {end_time}, total: {total_time}")

    Global_state.transcript_queue.put(None)
    gpu_thread = threading.Thread(target=gpu_worker, name="GPU_Worker - (THREAD)")
    gpu_thread.start()
    Global_state.transcript_queue.join()
    gpu_thread.join()
    Global_state.video_task_que.put(None)
    gpu_thread_offline = True
    worker_thread.join()
    log("Program completed successfully!")
    Montage_short_worker_thread = threading.Thread(target=Montage_short_worker,name="Montage_VideoCreation_worker")
    Montage_short_worker_thread.start()
    from Agents.Montage_agent import Run_short_montage_agent
    Run_short_montage_agent()
    Global_state.Montage_clip_task_Que.put(None)
    Montage_short_worker_thread.join()

