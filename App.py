import sys
import os
REPO_ROOT = os.path.abspath('.')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

GPEN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'GPEN'))
if GPEN_PATH not in sys.path:
    sys.path.insert(0, GPEN_PATH)
import GPEN.__init_paths
from GPEN.face_enhancement import FaceEnhancement
from smolagents import TransformersModel, VLLMModel, FinalAnswerTool, CodeAgent, tool,LiteLLMModel
from utility.Custom_Agent_Tools import SpeechToTextToolCUDA, SpeechToTextToolCPU, SpeechToText_short_creation_thread,Delete_rejected_line,SaveMotivationalText,create_motivationalshort ,Background_Audio_Decision_Model,open_work_file,montage_short_creation_tool
import gc
from neon.log import log
import yaml
import torchvision.transforms.functional as F
sys.modules['torchvision.transforms.functional_tensor'] = F
import torch.nn.functional as F
import subprocess
from moviepy.audio.fx import MultiplyVolume
from moviepy import VideoFileClip, ImageSequenceClip, TextClip, CompositeVideoClip,vfx,AudioFileClip,afx,CompositeAudioClip,afx
import cv2
import ffmpeg
from typing import List
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import onnxruntime as ort
from ultralytics.utils.ops import non_max_suppression
import pynvml
import time
import numpy as np
from utility.clean_memory import clean_get_gpu_memory
import torch
from moviepy.video.fx.CrossFadeIn import CrossFadeIn
from moviepy.video.fx.FadeIn import FadeIn
from moviepy.video.fx.FadeOut import FadeOut
import threading
import queue
import torch
from pydub import AudioSegment
import random
import re
from proglog import ProgressBarLogger
import utility.Global_state as Global_state
from queue import Queue
from dotenv import load_dotenv
from utility.create_montage_short import _create_montage_short_func, compose_montage_clips
import threading
import os

load_dotenv()

OPENAI_APIKEY = os.getenv("OPENAI_APIKEY")
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
log(f"Total Used: {mem_info.used/1e9:.1f}GB")

#---------------------------------------#
# Cleans all the log files Automatically
#---------------------------------------#
def Clean_log_onRun():
    with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\debug_performance\log.txt", "w", encoding="UTF-8") as w:
            w.write("")
    with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\debug_performance\VerifyAgentRun_data.txt", "w", encoding="UTF-8") as w:
            w.write("")
    with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\debug_performance\Token_logpath", "w", encoding="UTF-8") as w:
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



#---------------------------------#
#Global Variables
#----------------------------------#
count = 0
Video_count = 0
Global_model = None


Upload_YT_count = 0

#---------------------------------------#
# Queue / Threads / Functions /  Variables
#---------------------------------------#
def  clear_queue(q: Queue):
     with q.mutex:
          q.queue.clear()
          q.all_tasks_done.notify_all()
          q.unfinished_tasks = 0



#lage en klasse istedet mer ryddig....
#-----------------------------------#
# Reloading&Changing LLM-model
#-----------------------------------#
def Reload_and_change_model(model_name, message):
    global Global_model

    # Clean up existing model
    if Global_model:
        del Global_model
        clean_get_gpu_memory(threshold=0.6)

    log(message)

    if model_name == "gpt-4o":
        Global_state.set_current_global_model("gpt-4o")
        Global_model = LiteLLMModel(model_id="gpt-4o", api_key=OPENAI_APIKEY, temperature=0.0, max_tokens=16000)
        return Global_model

    elif model_name == "gpt-4-finetuned":
        Global_state.set_current_global_model("gpt-4-finetuned")
        Global_model = LiteLLMModel(
            model_id="ft:gpt-4.1-2025-04-14:personal-learnreflects:motivational-short-extractor:CHeRJbsB",
            api_key=OPENAI_APIKEY,
            max_tokens=16384
        )
        return Global_model

    elif model_name == "gpt-5":
        Global_state.set_current_global_model("gpt-5")
        Global_model = LiteLLMModel(
            model_id="gpt-5",
            reasoning_effort="high",
            api_key=OPENAI_APIKEY,
            max_tokens=20000
        )
        return Global_model
    elif model_name == "gpt-5-minimal":
        Global_state.set_current_global_model("gpt-5-minimal")
        Global_model = LiteLLMModel(
            model_id="gpt-5",
            reasoning_effort="minimal",
            api_key=OPENAI_APIKEY,
            max_tokens=20000
        )
        return Global_model
    elif model_name == "Phi-4-mini-instruct":
        Global_state.set_current_global_model("Phi-4-mini-instruct")
        Global_model = TransformersModel(
            model_id=r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen",
            device_map="auto",
            do_sample=False,
            temperature=0.0,
            max_new_tokens=10000,
            torch_dtype=torch.float16,
            load_in_4bit=True,
           # use_flash_attn=True
        )
        return Global_model

    else:
        log(f"Unknown model name: {model_name}. Available models: gpt-4o, gpt-4-finetuned, gpt-5")
        raise ValueError(f"Unsupported model name: {model_name}")

#---------------------------------#
# Classes
#----------------------------------#
class MyProgressLogger(ProgressBarLogger):
    def callback(self, **changes):
        for param, value in changes.items():
            log(f"{param}: {value}")
            log(f"{param}: {value}")







#-----------------------------------#
# Shorts Video Creation/Functions
#-----------------------------------#


def change_brightness(frame, amount=0.2):
    """
    Darken or brighten the frame.
    amount > 0 : brighten
    amount < 0 : darken
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 2] *= (1.0 + amount)
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
    changed_frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return changed_frame


def change_saturation(frame ,mode="Increase", amount=0.2):
     if mode == "Increase":
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1]  *= (1.0 + amount)
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        changed_frames = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
     elif mode == "Decrease":
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= (1.0 - amount)
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        changed_frames = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

     return changed_frames





def mix_audio(original_audio, background_music_path, bg_music_volume=0.25):
        bg_music = AudioFileClip(background_music_path)
        if bg_music.duration < original_audio.duration:
            bg_music = afx.audio_loop(bg_music, duration=original_audio.duration)
        else:
            bg_music = bg_music.subclipped(0, original_audio.duration)
        bg_music = bg_music.with_effects([MultiplyVolume(bg_music_volume)])
        original_audio = original_audio.with_effects([MultiplyVolume(1.0)])
        mixed_audio = CompositeAudioClip([original_audio, bg_music])
        log(f"[mix_audio] --> (original_audio.duration): {original_audio.duration}, (bg_music.duration): {bg_music.duration}, (mixed_audio.duration): {mixed_audio.duration}")
        return mixed_audio




def parse_editing_notes(notes):
        fade_in_duration = 0
        volume_reduction = 1.0
        import re

        fade_in_match = re.search(r'Fade in (?over|at) (\d+\.?\d*) seconds?', notes, re.IGNORECASE)
        if fade_in_match:
            fade_in_duration = float(fade_in_match.group(1))

        volume_match = re.search(r'lower volume by (\d+\.?+\d)%?', notes, re.IGNORECASE)
        if volume_match:
              percentage = float(volume_match.group(1))
              volume_reduction = 1.0 - (percentage / 100.0)

        return  {"fade_in_duration": fade_in_duration, "volume_reduction": volume_reduction}

def load_cube_lut(path):
    lut_data = []
    size = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.upper().startswith("LUT_3D_SIZE"):
                size = int(line.split()[1])
            elif line[0].isdigit() or line[0] == "-":
                values = [float(v) for v in line.split()]
                lut_data.append(values)

    if size is None:
        raise ValueError("Invalid .cube file: missing LUT_3D_SIZE")

    lut = np.array(lut_data).reshape((size, size, size, 3))
    return lut



def mix_audio_with_effects(original_audio, bg_music_path, bg_music_volume=0.48, editing_notes=""):
        bg_music = AudioFileClip(bg_music_path)


        if bg_music.duration < original_audio.duration:
            bg_music = afx.audio_loop(bg_music, duration=original_audio.duration)
        else:
            bg_music = bg_music.subclipped(0, original_audio.duration)


        effects = parse_editing_notes(editing_notes)
        fade_in_duration = effects["fade_in_duration"]
        volume_reduction = effects["volume_reduction"]
        time_specific = effects["time_specific"]


        if fade_in_duration > 0:
            bg_music = bg_music.fx(vfx.FadeIn, duration=fade_in_duration)


        if time_specific:
            print(f"Warning: Time-specific ducking requested ({time_specific}), but no timestamps provided. Applying uniform volume reduction.")
            bg_music = bg_music.volumex(bg_music_volume * volume_reduction)
        else:
            bg_music = bg_music.volumex(bg_music_volume * volume_reduction)

        original_audio = original_audio.volumex(1.0)

        mixed_audio = CompositeAudioClip([original_audio, bg_music])
        return mixed_audio




def detect_and_crop_frames_batch(frames, batch_size=8):
        TARGET_W, TARGET_H = 1080, 1920
        alpha = 0.1
        prev_cx, prev_cy = None, None
        cropped_frames = []
        onnx_path_gpu = r"c:\Users\didri\Desktop\LLM-models\Face-Detection-Models\yolov8x-face-lindevs_cuda.onnx"
        providers = ['CUDAExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 0
        session = ort.InferenceSession(onnx_path_gpu, sess_options, providers=providers)
        log(f"ONNX Runtime providers in use: {session.get_providers()}")
        input_name = session.get_inputs()[0].name
        total_batches = (len(frames) + batch_size - 1) // batch_size
        progress_bar = tqdm(total=len(frames), desc="[detect_and_crop_frames_batch]Processing frames", unit="frame", dynamic_ncols=True)
        try:
            for i in range(0, len(frames), batch_size):
                log(f"batch: {i} - {i + batch_size}")
                batch = frames[i:i+batch_size]
                original_count = len(batch)


                if len(batch) < batch_size:
                    pad_count = batch_size - len(batch)
                    batch += [np.zeros_like(batch[0])] * pad_count


                processed_batch = []
                for frame in batch:
                    img = cv2.resize(frame, (928, 928))
                    img = img.astype(np.float32) / 255.0
                    processed_batch.append(img.transpose(2, 0, 1))
                    progress_bar.set_postfix({
                                            "GPU Mem": f"{torch.cuda.memory_allocated()/1e9:.1f}GB",
                                            "Batch Size": f"{len(batch[:original_count])}/{batch_size}"
                                        })
                    progress_bar.update(1)

                if len(processed_batch) < batch_size:
                    processed_batch += [np.zeros((3, 928, 928), dtype=np.float32)] * (batch_size - len(processed_batch))

                input_tensor = np.stack(processed_batch).astype(np.float32)


                outputs = session.run(None, {input_name: input_tensor})[0]


                predictions = torch.tensor(outputs[:original_count])
                detections = non_max_suppression(predictions, conf_thres=0.25, iou_thres=0.45)


                for idx, (frame, det) in enumerate(zip(batch[:original_count], detections)):
                    h, w = frame.shape[:2]

                    if det is not None and len(det):

                        areas = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                        max_idx = torch.argmax(areas)
                        x1, y1, x2, y2 = det[max_idx, :4].cpu().numpy().astype(int)


                        x1 = int(x1 * w / 928)
                        y1 = int(y1 * h / 928)
                        x2 = int(x2 * w / 928)
                        y2 = int(y2 * h / 928)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    else:
                        cx, cy = w // 2, h // 2


                    if prev_cx is None or prev_cy is None:
                        sx, sy = cx, cy
                    else:
                        sx = int(alpha * cx + (1 - alpha) * prev_cx)
                        sy = int(alpha * cy + (1 - alpha) * prev_cy)
                    prev_cx, prev_cy = sx, sy


                    aspect_ratio = TARGET_W / TARGET_H
                    if w / h > aspect_ratio:
                        crop_h = h
                        crop_w = int(h * aspect_ratio)
                    else:
                        crop_w = w
                        crop_h = int(w / aspect_ratio)

                    x0 = max(0, min(sx - crop_w // 2, w - crop_w))
                    y0 = max(0, min(sy - crop_h // 2, h - crop_h))

                    cropped_frame = frame[y0:y0+crop_h, x0:x0+crop_w]
                    if cropped_frame.shape[:2] != (TARGET_H, TARGET_W):
                        cropped_frame = cv2.resize(cropped_frame, (TARGET_W, TARGET_H))
                    cropped_frames.append(cropped_frame)
        finally:
            progress_bar.close()
            del  predictions, detections, frame, det, h, w, areas, max_idx
            if batch is not None:
                del batch
            session = None
            clean_get_gpu_memory(threshold=0.8)
        return cropped_frames






#------------------------------------------------------------------------------------------------------------------------#
# create_short_video --> Function takes (video, start time/end time for video, video name, subtitles for video) as input
#------------------------------------------------------------------------------------------------------------------------#
def create_short_video(video_path, audio_path, start_time, end_time, video_name, subtitle_text,Video_output_path=None):
    YT_channel = Global_state.get_current_yt_channel()
    log(f"YT_channel: {YT_channel}")
    probe = ffmpeg.probe(video_path)
    log(probe)
    format_info = probe.get('format', {})
    bitrate = int(format_info.get('bit_rate', 0))
    video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
    video_codec = video_streams[0]['codec_name'] if video_streams else None
    audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']
    audio_codec = audio_streams[0]['codec_name'] if audio_streams else None

    def group_subtitle_words_in_triplets(subtitle_words):
         chunks = []
         i = 0
         offset = float(subtitle_words[0]['start']) if subtitle_words else 0.0
         while i < len(subtitle_words):
              triplet = subtitle_words[i:i+4]
              if not triplet:
                  break

              text_chunk = ''.join([w['word'].strip() + ' ' for w in triplet]).strip().upper()
              start = float(triplet[0]['start']) - offset
              end = float(triplet[-1]['end']) - offset
              duration = end - start
              start = max(0.0, start)
              duration = max(0.0, duration)

              chunks.append({'text': text_chunk, 'start': start, 'end': end, 'duration': duration})
              i += 4
         return chunks

    try:
       triplets = group_subtitle_words_in_triplets(subtitle_text)
    except Exception as e:
         log(f"[group_subtitle_words_in_triplets] Error during grouping of subtitles in triplets. {str(e)} ")


    def create_subtitles_from_triplets(triplets):
        from moviepy.video.fx import FadeIn, FadeOut
        text_clips = []

        for i, c in enumerate(triplets):


            txt_clip = TextClip(
                text=c['text'],
                font=r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Utils-Video_creation\Fonts\OpenSans-VariableFont_wdth,wght.ttf",
                font_size=55,
                margin=(10, 10),
                text_align="center",
                vertical_align="center",
                horizontal_align="center",
                color='white',
                stroke_color="black",
                stroke_width=3,
                size=(1000, 300),
                method="label",
                duration=c['duration']
               ).with_start(c['start']).with_position(('center', 0.50), relative=True)


            text_clips.append(txt_clip)
            log(f"Subtitle clip {i}: text='{c['text'][:20]}...', total_duration={c['duration']:.3f}s")
        return text_clips





    #------------------------------------------------------------------------#
    # CREATES THE (START & END) OF the currentclip from FULL ORIGINAL VIDEO
    #------------------------------------------------------------------------#
    full_video = VideoFileClip(video_path)
    clip = full_video.subclipped(start_time, end_time)
    log(f"clip duration: {clip.duration}, clip fps: {clip.fps}, clip width: {clip.w}, clip height: {clip.h}, start_time: {start_time}, end_time: {end_time}, video_path: {video_path}")
    log(f"Clip fps: {clip.fps}")




#--------------------------------------------------#
# Extrcting frames from original video to a LIST
#--------------------------------------------------#
    frames = []
    for  frame in clip.iter_frames():
        frames.append(frame)
        frame_height, frame_width = frame.shape[:2]
    log(f"[Extracting original video frames]  1. frames: {len(frames)} frames.\n [CLIP.ITER] Height: {frame_height}, Width: {frame_width}")






# --------------------------------#
# Yolo8/facedetection + Cropping
# --------------------------------#
    clean_get_gpu_memory(threshold=0.1)
    cropped_frames = detect_and_crop_frames_batch(frames=frames,batch_size=8)
    log(f"2. cropped frames length: {len(cropped_frames)}")
    del frames

    apply_lut = random.choice([True, False])
# --------------------------------#
# Appyling LUT to frames
# --------------------------------#
    # if apply_lut:
    #     from apply_lut_to_image import apply_lut_to_frames
    #     frames_with_lut  = apply_lut_to_frames(cropped_frames, lut_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Utils-Video_creation\LUT\Black & white cube\blackwhite1.cube")


    # frames_for_blender = frames_with_lut if apply_lut else cropped_frames
# ------------------------------------#
#  Blender Tweaking frames with details
# ------------------------------------#
    # from blender import enhance_frames_bpy
    # blender_frames = enhance_frames_bpy(frames_for_blender, batch_size=8)
    # del frames_for_blender



# -----------------#
# color/Adjustment
# -----------------#
    # log(f"\n\n[COLOR ADJUSTMENT] PROCCESS starting...")
    # try:
    #     Color_corrected_frames = [change_saturation(frame,mode="Increase", amount=0.15) for frame in blender_frames]
    #     log(f"3. Color_corrected_frames length: {len(Color_corrected_frames)}")
    #     del blender_frames
    # except Exception as e:
    #      log(f"Error during color correction: {str(e)}")






    # try:
    #     subtitle_clips = create_subtitles_from_triplets(triplets)
    # except Exception as e:
    #         log(f"error during [create_subtitles_from_triplets]: {str(e)}")


    # try:
    #     log(f"\n\n[SUBTITLE PROCESSING] Adding neon subtitles to frames...")
    #     from neon.build_random_mode_pairs_from_words import build_random_mode_pairs_from_words
    #     mode_name = "0-2-0-word-by-word"
    #     frames_with_subtitles = build_random_mode_pairs_from_words(cropped_frames,subtitle_text, clip.duration, clip.fps, mode_name=mode_name)
    #    # del Color_corrected_frames
    #     log(f"4. Frames with subtitles length: {len(frames_with_subtitles)}")
    # except Exception as e:
    #     log(f"Error during subtitle processing: {str(e)}")
    #     return

# # ----------------------#
# #   FACEENCHANCEMENT
# # ----------------------#
#     class Face_enchance_Args:
#             model = 'GPEN-BFR-2048'
#             task = 'FaceEnhancement'
#             key = None
#             in_size = 2048
#             out_size = 2048
#             channel_multiplier = 2
#             narrow = 1
#             alpha = 0.5
#             use_sr = True
#             use_cuda = True
#             save_face = False
#             aligned = False
#             sr_model = 'realesrnet'
#             sr_scale = 2
#             tile_size = 0
#             ext = '.png'

#     log(f"\n\n[FACEENCHACEMENT] PROCCESS starting...")
#     Skin_texture_enchancement = FaceEnhancement(
#          Face_enchance_Args,
#         in_size=Face_enchance_Args.in_size,
#         model=Face_enchance_Args.model,
#         use_sr=Face_enchance_Args.use_sr,
#         device='cuda' if Face_enchance_Args.use_cuda else 'cpu'
#     )
#     FaceEnhancement_frames = []
#     try:
#          for f_frames in tqdm(Color_corrected_frames, desc="[FaceEnhancement]  proccessing frames", unit="frame"):
#               log(f"Input frame size: {f_frames.shape[1]} x {f_frames.shape[0]}")
#               frame_height, frame_width = f_frames.shape[:2]
#               frame_bgr = cv2.cvtColor(f_frames, cv2.COLOR_RGB2BGR)
#               enchanced_frame_bgr, _, _ = Skin_texture_enchancement.process(frame_bgr)
#               RGB_face_enchanced_frame = cv2.cvtColor(enchanced_frame_bgr, cv2.COLOR_BGR2RGB)
#               FaceEnhancement_frames.append(RGB_face_enchanced_frame)
#               log(f"Enhanced frame size: {enchanced_frame_bgr.shape[1]} x {enchanced_frame_bgr.shape[0]}")
#          del Color_corrected_frames
#          log(f"[FaceEnhancement] 5. FaceEnhancement_frames length: {len(FaceEnhancement_frames)}")


#          log(f"Cleared cache and collected garbage")
#     except Exception as e:
#             log(f"[FaceEnhancement] Error: {str(e)}")






#-------------------------------#
# Creating videoclip from frames
#-------------------------------#
    try:
       processed_clip = ImageSequenceClip(cropped_frames, fps=clip.fps).with_duration(clip.duration)
       gc.collect()
    except Exception as e:
         log(f"[processed_clip] ERROR: {str(e)}")
    clean_get_gpu_memory(threshold=0.3)
#    del FaceEnhancement_frames



#-----------------------------------------------#
#     Adds Logo/overlay for YT_channel
#-----------------------------------------------#
    YT_channel = None

    try:
        global Upload_YT_count
        log(f"current YT_count: {Upload_YT_count}")
        with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\Youtube_Upload_folder\latest_uploaded.txt","r", encoding="UTF-8") as r:
            Latest_Yt_channel = r.read().strip()
            log(f"Latest_Yt_channel: {Latest_Yt_channel}")
            YT_channel = Latest_Yt_channel


        if Latest_Yt_channel == "MR_Youtube":
             YT_channel = "LR_Youtube"
             overlay_clip = VideoFileClip(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\alpha_mask\YT_logo\LOGO_LR.mp4",has_mask=True)
             log(f"YT_channel: {YT_channel}")
        elif Latest_Yt_channel == "LR_Youtube":
             YT_channel = "LRS_Youtube"
             overlay_clip = VideoFileClip(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\alpha_mask\YT_logo\LOGO_LRS.mp4",has_mask=True)
             log(f"YT_channel: {YT_channel}")
        elif Latest_Yt_channel == "LRS_Youtube":
             YT_channel = "LM_Youtube"
             overlay_clip = VideoFileClip(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\alpha_mask\YT_logo\LOGO_LM.mp4",has_mask=True)
             log(f"YT_channel: {YT_channel}")
        elif Latest_Yt_channel == "LM_Youtube":
             YT_channel = "MR_Youtube"
             overlay_clip = VideoFileClip(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\alpha_mask\YT_logo\LOGO_MR.mp4",has_mask=True)
             log(f"YT_channel: {YT_channel}")

        Global_state.set_current_yt_channel(YT_channel)

        with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\Youtube_Upload_folder\latest_uploaded.txt","w", encoding="UTF-8") as w:
            w.write(YT_channel)

        overlay_clip = overlay_clip.subclipped(0, clip.duration)
        logo_mask = overlay_clip.to_mask()
        logo_with_mask = overlay_clip.with_mask(logo_mask)


        if Upload_YT_count == 4:
            Upload_YT_count = 0

    except Exception as e:
        log(f"Error during: finalizing clip with subtitle_clips:  {str(e)}")


#-----------------------------------------------#
# Adds Subtitles + Logo to the video
#-----------------------------------------------#
    final_clip = CompositeVideoClip(
                [processed_clip.with_position('center')]  +  [logo_with_mask.with_position('center',0.50)] ,
                size=processed_clip.size
                )

    del logo_with_mask


    fade = CrossFadeIn(1.5)
    final_clip = fade.apply(final_clip)

# -------------------------------#
# Adds Background Music to video
# -------------------------------#
    log("#######choosing Background Audio########\n")
    try:
        already_uploaded_videos = f"Video_clips/Youtube_Upload_folder/{YT_channel}/already_uploaded.txt"

        result = Background_Audio_Decision_Model(audio_file=audio_path,video_path=video_path,already_uploaded_videos=already_uploaded_videos)
        if os.path.exists(audio_path):
            os.remove(audio_path)
            log(f"Removed temporary audio file: {audio_path}")

        background_audio = result.get("path", "")
        song_name = result.get("song_name", "")

        log(f"Selected background audio: {song_name}")
        Background_Audio_Reason = result.get("reason", "")
        print(f"Background_Audio_Reason: {Background_Audio_Reason}")
    except Exception as e:
         log(f"error during: [Background_Audio_Decision_Model]: {str(e)}")


    log(f"Background audio path: {background_audio} \n Reason: {Background_Audio_Reason}\n")

    if background_audio:
        background_music_path = background_audio
        final_clip.audio = mix_audio(clip.audio, background_music_path, bg_music_volume=0.4)
    else:
        final_clip.audio = clip.audio
        log(f"keeping original audio")
        Background_Audio_Reason = "original audio only"

    log(f"(AUDIO DURATION): {final_clip.audio.duration}")









#-----------------------------------------------------#
#    Adds fade in/out to the video and sets the FPS
#------------------------------------------------------#
    final_clip = FadeIn(duration=0.1).apply(final_clip)
    final_clip = FadeOut(duration=0.1).apply(final_clip)
    final_clip.fps = clip.fps



#-----------------------------------------------------#
#    Cleans cpu/gpu memory
#------------------------------------------------------#
    clean_get_gpu_memory(threshold=0.2)


#-----------------------------------------------------#
#    Writes the final Video
#------------------------------------------------------#
    output_dir = f"./Video_clips/Youtube_Upload_folder/{YT_channel}"
    os.makedirs(output_dir, exist_ok=True)

    if Video_output_path:
         out_path = Video_output_path
    else:
        out_path = os.path.join(output_dir, f"{video_name}.mp4")


    _finalclipduration = final_clip.duration
    log(f"FINAL CLIP DURATION: {_finalclipduration}")

    final_clip.write_videofile(
    out_path,
    logger='bar',
    codec="libx264",
    preset="slow",
    audio_codec="aac",
    threads=8,

    ffmpeg_params=[
        "-crf", "8",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-ar", "48000",
        "-vf", "minterpolate=fps=30",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        "-movflags", "+faststart",
    ],
    audio_bitrate="384k",
    remove_temp=True
        )


    full_video.close()
    clip.close()
    clean_get_gpu_memory(threshold=0.1)








#-----------------------------------------------------#
#    Boosts x2 FPS on full video
#------------------------------------------------------#
    from utility.RIFE_FPS import run_rife
    try:
      output_video = run_rife(out_path)
      log(f"video is completed: output path : {out_path}, video name: {video_name} video_fps: {clip.fps}, codec: {video_codec}, bitrate: {bitrate}, audio_codec: {audio_codec}, subtitles: {subtitle_text} \n Final video resolution (width x height): {final_clip.size[0]} x {final_clip.size[1]}")
      log(f"Interpolated video saved to: {output_video}")
    except Exception as e:
         log(f"[run_rife] ERROR: {str(e)}")
    finally:
        os.remove(out_path)
        clean_get_gpu_memory(threshold=0.3)





#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# A Agent creates optimized (Title, Description, Hashtags, Tags, category, publishAt) after analyzing similar trending videos related to input video &  Uploads the video to Youtube
# - Reloads the Global Model
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    try:
        global Global_model
        if 'Global_model' not in globals():
                Global_model = None
        from Agent_AutoUpload.upload_Socialmedia import upload_video
        log(f"\n\n\n\n\n\n\n\n\n\n-------------------------------------------------------------------\n\n\n\n")
        log("Youtube uploading & agent STARTING....")

        try:
            Current_model_loaded = Global_state.get_current_global_model()
            if Global_model and Current_model_loaded == "gpt-4o":
                del Global_model
                Global_model = None
                clean_get_gpu_memory(threshold=0.3)
        except NameError:
                pass


        if Global_model is None:
             try:
                Global_model = Reload_and_change_model(model_name="gpt-5", message="Reloading model to -> gpt-5 before running [upload_video]")
             except Exception as e:
                  log(f"Error reloading and changing model to gpt-5: {str(e)}")

        if Global_model is None:
            raise ValueError("Failed to initialize Global_model for upload_video")


        YT_channel = Global_state.get_current_yt_channel()

        try:
            social_media = upload_video(model=Global_model,file_path=output_video,subtitle_text=subtitle_text,YT_channel=YT_channel,background_audio_=Background_Audio_Reason,song_name=song_name,video_duration=_finalclipduration)
        except Exception as e:
            log(f"error during upload_video: {str(e)}")
        log(f"Done with uploading to {social_media}")
    except Exception as e:
         log(f"error during uploading: {str(e)}")

    finally:
         clean_get_gpu_memory(threshold=0.8)
        #  if 'Global_model' in globals() and Global_model is not None:
        #       del Global_model












def truncate_audio(audio_path, start_time, end_time, output_path):
            """
            Truncate an audio file from start_time to end_time and save it to output_path.

            start_time, end_time: in seconds (float or int)
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

                start_sample = round(start_time * frame_rate )
                end_sample = round(end_time * frame_rate )
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

            result = tool.forward({"audio": audio_for_clip,"subtitle_text": subtitle_text, "original_start_time": video_start_time, "original_end_time:": video_end_time})

            crafted_Subtitle_text = result["matched_words"]
            new_video_start_time = float(result["video_start_time"])
            new_video_end_time = float(result["video_end_time"])
            log(f"[run_video_short_creation_thread] creating video now... \n start_time: {new_video_start_time} \n end_time: {new_video_end_time},  \n subtitle_text: {crafted_Subtitle_text}")

            if Video_Title_name is None:
                Video_title = "short1" + str(current_count)
            else:
                 Video_title = Video_Title_name


            create_short_video(video_path=text_video_path, audio_path=audio_for_clip, start_time=new_video_start_time, end_time = new_video_end_time, video_name = Video_title, subtitle_text=crafted_Subtitle_text,Video_output_path=Video_output_path)
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

             video_url,audio_path, final_start_time, final_end_time, subtitle_text = item

             log(f"current work being proccessed/Retrieved WORK: {video_url}\naudio_path:{audio_path}\n {final_start_time}\n {final_end_time}\n {subtitle_text}\n")
          except queue.Empty:
                log(f"[video_creation_worker] Queue empty, waiting for new tasks...")
                continue
          try:
                run_video_short_creation_thread(video_url,audio_path,final_start_time, final_end_time, subtitle_text)
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
def wait_for_proccessed_video_complete(queue: Queue, check_interval=30):
    """Blocks until the queue is empty, checking every `check_interval` seconds."""
    log(f"\n\n\n\n\n\n[wait_for_proccessed_video_complete]")
    while not queue.empty():
          log(f"[wait_for_proccessed_video_complete]  waiting for video_task_que to be empty: items remaining: {queue.qsize()}")
          time.sleep(check_interval)
    log("[wait_for_proccessed_video_complete]✅ video_task_que is now empty!!!")



#montage
def Montage_short_worker():
     montage_count = 0
     # Aggregator for collecting parts per montage group (N)
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

               video_path, audio_path, start_time, end_time, subtitle_text, order, montage_id, YT_channel = item
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
               part_basename = f"{group_key}_{order}.mp4"
               part_output_path = os.path.join(parts_dir, part_basename)
               log(f"[Montage_short_worker] Rendering part group={group_key} order={order} -> {part_output_path}")

               # Render the individual part (this writes part_output_path, then creates RIFE file and deletes original)
               _create_montage_short_func(
                   video_path=video_path,
                   start_time=start_time,
                   end_time=end_time,
                   subtitle_text=subtitle_text,
                   video_name=os.path.splitext(part_basename)[0],
                   Video_output_path=part_output_path,
                   YT_channel=YT_channel,
               )

               # After RIFE, the final file is "<stem>_rife.mp4"
               rife_output_path = os.path.splitext(part_output_path)[0] + "_rife.mp4"
               final_part_path = rife_output_path if os.path.exists(rife_output_path) else part_output_path
               log(f"[Montage_short_worker] Part ready: {final_part_path}")

               # Update jobs aggregator (store paths + channel per group)
               ready_to_compose = False
               group_channel = None
               with jobs_lock:
                   group = jobs.setdefault(group_key, {"paths": {k: None for k in orders_required}, "yt": None})
                   # Lock channel to the first item; warn if mismatch later
                   if group["yt"] is None:
                       group["yt"] = YT_channel
                   elif group["yt"] != YT_channel:
                       log(f"[Montage_short_worker][WARN] YT_channel mismatch within group {group_key}: stored={group['yt']} incoming={YT_channel}. Using stored.")
                   group["paths"][order] = final_part_path
                   if all(group["paths"][k] for k in orders_required):
                       ready_to_compose = True
                       group_channel = group["yt"]

               # Compose when all three parts are ready
               if ready_to_compose:
                   try:
                       montage_count += 1
                       ordered_paths = [jobs[group_key]["paths"][k] for k in orders_required]
                       final_output_dir = os.path.join(".", "Video_clips", "Montage_clips")
                       os.makedirs(final_output_dir, exist_ok=True)
                       final_output_path = os.path.join(final_output_dir, f"montage_{group_key}.mp4")
                       log(f"[Montage_short_worker] Composing montage group={group_key} -> {final_output_path}")
                       output_path = compose_montage_clips(ordered_paths, final_output_path)
                       from Agent_AutoUpload.upload_Socialmedia import upload_MontageClip
                       log(f"[Montage_short_worker] Uploading montage to YT channel: {group_channel}")
                       model = Reload_and_change_model("gpt-4o",message="reloaded model inside Montage_short_worker before upload_MontageClip")
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




#-------------------------------------------------------------------------------------------#
#           Writes Information/logging outputs from Agent Runs to a text file.
#-------------------------------------------------------------------------------------------#
def save_full_io_to_file(modelname: str, input_chunk: str, reasoning_steps: list[str], model_response: str, file_path: str) -> None:
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




#-------------------------------------------------------------------------------------------------#
# Agent that verifies that a Quote saved by (transcript_reasoning_Agent) is indeed a valid Quote
#-------------------------------------------------------------------------------------------------#
def verify_saved_text_agent(agent_saving_path):
    global Global_model
    if Global_model is None or Global_state.get_current_global_model() != "gpt-5":
        Global_model = Reload_and_change_model("gpt-5",message="reloaded model inside verify_saved_text_agent")

    with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Prompt_templates\verify_agent_system_prompt.yaml", "r", encoding="utf-8") as f:
         verify_system_prompt = yaml.safe_load(f)

    final_answer = FinalAnswerTool()
    create_motivational_short_agent = CodeAgent(
        model=Global_model,
        tools=[create_motivationalshort,Delete_rejected_line,final_answer],
        max_steps=1,
        prompt_templates=verify_system_prompt,
        verbosity_level=1
    )

    with open(agent_saving_path, "r", encoding="utf-8") as f:
             saved_quotes_text = f.read()
             if not saved_quotes_text.strip():
                  Global_state.chunk_proccesed_event.set()
                  log(f"Agent_Saving_path is (EMPTY) \n - {agent_saving_path}")
                  return

    Blocks = re.findall(r"===START_TEXT===.*?===END_TEXT===", saved_quotes_text, re.DOTALL)

    chunk_size = 4
    chunks = [Blocks[i:i+chunk_size] for i in range(0, len(Blocks), chunk_size)]

    for idx, chunk in enumerate(chunks, 1):
         combined_text = "\n".join(chunk)

         task = f"Analyze all the textblocks. You must Reject those that are not valid for a motivational short using `Delete_rejected_line`.  You must  Approve those that are valid by passing them into `create_motivationalshort`.  Here is the text to analyze: {combined_text}"

         model_response = create_motivational_short_agent.run(task=task)
         print(model_response)
         clean_get_gpu_memory(threshold=0.8)
    with open(agent_saving_path, "w", encoding="utf-8") as f:
         f.write("")

    Global_state.chunk_proccesed_event.set()
    log("All chunks processed. File has been emptied.")
    del create_motivational_short_agent


#---------------------------------------------------------------------------------------------------------------------------------#
#   Agent that analyzes text  from transcript by reading it (chunk for chunk) --->  (saves Quote identified in podcast transcript.
#---------------------------------------------------------------------------------------------------------------------------------#
def Transcript_Reasoning_AGENT(transcript_path,agent_txt_saving_path):
    from utility.Custom_Agent_Tools import ChunkLimiterTool
    log(f"✅Transcript_Reasoning_AGENT (Running)")

    global Global_model
    if Global_model is None:
         Global_model = Reload_and_change_model("gpt-5",message="reloaded model inside Transcript_Reasoning_AGENT")

    loaded_reasoning_agent_prompts = r'C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Prompt_templates\structured_output_prompt_TranscriptReasoning_gpt5.yaml'
    with open(loaded_reasoning_agent_prompts, 'r', encoding='utf-8') as f:
            Prompt_template = yaml.safe_load(f)

    final_answer = FinalAnswerTool()
    Reasoning_Text_Agent = CodeAgent(
        model=Global_model,
        tools=[SaveMotivationalText,final_answer],
        max_steps=1,
        prompt_templates=Prompt_template,
        verbosity_level=1,
        use_structured_outputs_internally=True,

    )
    chunk_limiter = ChunkLimiterTool()

    reasoning_log = []
    chunk_limiter.reset()
    while True:
        reasoning_log.clear()


        chunk = chunk_limiter.forward(file_path=transcript_path, max_chars=6000)

        if not chunk.strip():
                log(f"\nTranscript Path is (EMPTY)\n -{transcript_path}")
                del Reasoning_Text_Agent
                clean_get_gpu_memory(threshold=0.2)
                _transcript = ""
                with open(agent_txt_saving_path, "r", encoding="utf-8") as r:
                   _transcript = r.read()
                   if  _transcript.strip():
                        with open(Global_state._agent_saving_copy_path, "w", encoding="utf-8") as w:
                            w.write(_transcript)
                            log(f"Copied Saved text:\n - {_transcript}\n by (transcript_reasoning_agent) to:\n - {Global_state._agent_saving_copy_path}")


                clean_get_gpu_memory(threshold=0.8)
                verify_saved_text_agent(agent_txt_saving_path)
                log(f"verify_saved_text_agent is done, exited inside transcript agent.")
                wait_for_proccessed_video_complete(Global_state.video_task_que)
                Global_state.chunk_proccesed_event.clear()  # Reset for next batch
                log(f"done with work. exiting transcript reasoning agent to retrieve the next items from the queue.")
                break

        task = f"""
                Your task is to Identify Qualifying Motivational Texts & Save them if any is found in the chunk.
                Here is the chunk you must analyze
                [chunk start]\n
                {chunk}\n
                [chunk end]
                """

        result = Reasoning_Text_Agent.run(
                    task=task,
                    additional_args={"text_file": agent_txt_saving_path},
                )

        log(f"[Path to where the [1. reasoning agent ] saves the motivational quotes  ]: {agent_txt_saving_path}")
        log(f"Agent response: {result}\n")
        chunk_limiter.called = False
        clean_get_gpu_memory(threshold=0.8)


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
            "-af", "afftdn=nr=20:nf=-30:tn=1",
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

    Reload_and_change_model(model_name="gpt-5",message="Loading up model inside GPU worker")

    log(f"Loaded Global_model on device")
    while True:
        item = Global_state.transcript_queue.get()
        if item is None:
            log("[Shutdown signal received] - exiting GPU worker")
            Global_state.transcript_queue.task_done()
            break

        video_path_url, transcript_text_path,agent_txt_saving_path,_audio_path = item


        Global_state.set_current_audio_path(_audio_path)
        Global_state.set_current_videourl(video_path_url)
        Global_state.set_current_textfile(agent_txt_saving_path)
        folder = os.path.dirname(agent_txt_saving_path)
        copy_agent_txt = "agent_saving_path_copy"
        Global_state._agent_saving_copy_path = os.path.join(folder,f"{copy_agent_txt}.txt")
        log(f"Item Retrieved from (Transcript Queue) ready for Processing: Global state:\n - set_current_videourl:{video_path_url}\n, Global state: set_current_audio_path:\n - {_audio_path}\n Transcript text path - {transcript_text_path}\n agent is saving text from transcript to Global state:\n set_current_textfile:{agent_txt_saving_path}")

        try:
            Transcript_Reasoning_AGENT(transcript_text_path, agent_txt_saving_path)
            log(f"Transcript_Reasoning_AGENT  (EXITED)")
        finally:
            log(f"Transcript Queue (TASK DONE)\n - {transcript_text_path}")
            Global_state.transcript_queue.task_done()
















#montage
def Run_short_montage_agent():
    global Global_model
    Global_model =  Reload_and_change_model("gpt-5",message="Reloading model to -> phi-4-FineTuned before running [Montage_short_agent]")
    with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Prompt_templates\system_prompt_structured_Shorts_combination_agent.yaml", "r", encoding="utf-8") as r:
         montage_agent_systemprompt = yaml.safe_load(r)

         Montage_task = """
            Your task is to create the best motivational montage shorts video.

            Input:
                - 4–5 text files, each representing a video.
                - Each video title section contains  “motivational snippets” (quotes/excerpts).

            Task Goal:
                - Combine content from each video into a single, cohesive, and motivating script (for a short-form video).

            Workflow:
            1. Analyze snippets in each video
                - Read all snippets per file.
                - Understand the core message and tone (e.g., perseverance, growth, overcoming fear, discipline).
                - Consider merges or connection for a strong punshy motivational short montage.

            2. Select compatible content from each Video title
                - You may choose:
                    * A full snippet, OR
                    * One or more complete sentences from within a snippet (only if the sentence has more than 4 words).
                - Chosen content must naturally fit together.
                - Ensure smooth flow, so when played in sequence, it sounds like one motivational speech.
                - Avoid combinations that feel disjointed or contradictory.
                - Make sure that when the content from each video title is composed. It does not exceed the length of 30 seconds.

            3. Decide the sequence
                - Arrange chosen content from each Video Title in a logical order:
                    * Opening: something that hooks attention or sets the theme.
                    * Middle: a challenge, reflection, or message of resilience.
                    * Ending: a powerful punchline, encouragement, or call-to-action.

            4. Generate the output
                - Deliver 3-5 `montage_short_creation_tool` tool calls.
                - Make sure it can be read aloud smoothly and feels like a finished motivational speech.

            """
    agent = CodeAgent(
         model=Global_model,
         verbosity_level=1,
         max_steps=3,
         prompt_templates=montage_agent_systemprompt,
         tools=[montage_short_creation_tool,open_work_file],
         use_structured_outputs_internally=True
        )
    additional_args = {
         "work_queue_folder": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\work_queue_folder"
         }


    response = agent.run(task=Montage_task, additional_args=additional_args)
    log(f"Montage Agent response: {response}")
    del agent



if __name__ == "__main__":
    clean_get_gpu_memory(threshold=0.1)
    Clean_log_onRun()

    worker_thread = threading.Thread(target=video_creation_worker,name="Video_creation(THREAD)")
    worker_thread.start()

    video_paths = [
              r"c:\Users\didri\Documents\Hardship Is An Opportunity To Improve - Bugzy Malone (4K) [yHX6OFGivWo] [1080p].mp4",
              r"c:\Users\didri\Documents\How To Find Direction When Nothing Feels Right - Chris Bumstead (4K) [SO155Z0mrc4] [1080p].mp4",
              r"c:\Users\didri\Documents\The Endless Pursuit of Progress - Sam Sulek (4K) [5117cPLuqB0] [1080p].mp4",
              r"c:\Users\didri\Documents\What to do When You’ve Lost Purpose in Life - Chris Bumstead [MNTC1P55JpA] [1080p].mp4"
              r"c:\Users\didri\Documents\The Art of Living a Courageous Life - Matthew McConaughey (4K) [y_woFP79F0Q] [1080p].mp4",
    ]
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
    worker_thread.join()
    log("Program completed successfully!")
    # Montage_short_worker_thread = threading.Thread(target=Montage_short_worker,name="Montage_VideoCreation_worker")
    # Montage_short_worker_thread.start()
    # Run_short_montage_agent()
    # Global_state.Montage_clip_task_Que.put(None)
    # Montage_short_worker_thread.join()

