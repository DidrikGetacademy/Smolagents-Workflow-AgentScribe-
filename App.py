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
from smolagents import TransformersModel, FinalAnswerTool, CodeAgent, tool
from Custom_Agent_Tools import SpeechToTextToolCUDA, SpeechToTextToolCPU, SpeechToText_short_creation_thread,Delete_rejected_line,SaveMotivationalText,create_motivationalshort
import tempfile
import gc
from log import log
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
import cv2
import onnxruntime as ort
from ultralytics.utils.ops import non_max_suppression
import pynvml
import time
import numpy as np
import cv2
from clean_memory import clean_get_gpu_memory
import torch
from moviepy.video.fx.FadeIn import FadeIn
from moviepy.video.fx.FadeOut import FadeOut
import threading
import queue
import torch
from pydub import AudioSegment
import datetime
import re 
from proglog import ProgressBarLogger
import Global_state
from queue import Queue
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
ActiveProgress=False
Upload_YT_count = 0  

#---------------------------------------#
# Queue / Threads / Functions /  Variables
#---------------------------------------#
def  clear_queue(q: Queue):
     with q.mutex:
          q.queue.clear()
          q.all_tasks_done.notify_all()
          q.unfinished_tasks = 0




#-----------------------------------#
# Reloading&Changing LLM-model
#-----------------------------------#
def Reload_and_change_model(model):
     if model == "Qwen7b":
            model = TransformersModel(
            model_id = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen\Qwen2.5-Coder-7B-Instruct",
                load_in_4bit=True,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype="auto",
                do_sample=False,
                use_flash_attn=True
            )
            return model
     elif model == "phi-4":
            global Global_model
            Global_model = TransformersModel(
                    model_id = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\microsoft\Merged_model",
                    load_in_4bit=True,
                    device_map="auto",
                    torch_dtype="auto",
                    do_sample=False,
                    use_flash_attn=True     
                    )
            return Global_model
          


#---------------------------------#
# Classes
#----------------------------------#
class MyProgressLogger(ProgressBarLogger):
    def callback(self, **changes):
        for param, value in changes.items():
            log(f"{param}: {value}")
            log(f"{param}: {value}")




class Face_enchance_Args:
        model = 'GPEN-BFR-512'
        task = 'FaceEnhancement'
        key = None
        in_size = 512
        out_size = None
        channel_multiplier = 2
        narrow = 1
        alpha = 0.5
        use_sr = False
        use_cuda = True
        save_face = False
        aligned = False
        sr_model = 'realesrnet'
        sr_scale = 1
        tile_size = 0
        ext = '.jpg'





#-----------------------------------#
# Shorts Video Creation/Functions
#-----------------------------------#
def change_saturation(frame ,mode="Increase", amount=0.2):
     if mode == "Increase":
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1]  *= (1.0 + amount)
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        changed_frames = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
     elif mode == "Decrease": 
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = 0  
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        changed_frames = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
          
     return changed_frames


def enhance_detail_and_sharpness(frame_bgr, clarity_factor=0.2, sharpen_amount=0.2):
    """
    Kombinerer detail layer clarity + mild sharpen på ett bilde.
    
    Args:
        frame_bgr: Inngangsbilde (BGR)
        clarity_factor: Hvor sterkt detail layer boostes (0.0–2.0)
        sharpen_amount: Hvor mye mild sharpen (0.0–2.0)
    Returns:
        Forbedret bilde (BGR)
    """

    smooth = cv2.bilateralFilter(frame_bgr, d=9, sigmaColor=75, sigmaSpace=75)


    detail_layer = cv2.subtract(frame_bgr, smooth)


    boosted_detail = cv2.addWeighted(detail_layer, clarity_factor, detail_layer, 0, 0)

    clarity_frame = cv2.add(frame_bgr, boosted_detail)

    blur = cv2.GaussianBlur(clarity_frame, (0, 0), 3)
    sharpened = cv2.addWeighted(clarity_frame, 1.0 + sharpen_amount, blur, -sharpen_amount, 0)

    return sharpened

def sharpen_frame_naturally(frame_bgr):
            from PIL import ImageFilter,Image
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            sharpned_pil = pil_img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=90,threshold=3))
            sharpned_rgb = np.array(sharpned_pil)
            sharpened_bgr = cv2.cvtColor(sharpned_rgb, cv2.COLOR_RGB2BGR)
            log(f"sharpening completed")
            return sharpened_bgr



def mix_audio(original_audio, background_music_path, bg_music_volume=0.15):
        bg_music = AudioFileClip(background_music_path)

        if bg_music.duration < original_audio.duration:
            bg_music = afx.audio_loop(bg_music, duration=original_audio.duration)
        else:
            bg_music = bg_music.subclipped(0, original_audio.duration)

        bg_music = bg_music.with_effects([MultiplyVolume(bg_music_volume)])
        original_audio = original_audio.with_effects([MultiplyVolume(1.0)])

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
                    log(f"processed batch: {len(processed_batch)} of {total_batches}")
            
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
                    log(f"[detect_and_crop_frames_batch] appended frame...")
                    cropped_frames.append(cropped_frame)
        finally:
            log(f"[detect_and_crop_frames_batch] Height: {TARGET_H}, Width: {TARGET_W}")
            progress_bar.close() 
            del  predictions, detections, frame, det, h, w, areas, max_idx
            if batch is not None:
                del batch
            session = None
            torch.cuda.empty_cache()
            gc.collect()
        return cropped_frames
    







#------------------------------------------------------------------------------------------------------------------------#
# create_short_video --> Function takes (video, start time/end time for video, video name, subtitles for video) as input
#------------------------------------------------------------------------------------------------------------------------#
def create_short_video(video_path, start_time, end_time, video_name, subtitle_text):
   # YT_channel = Global_state.get_current_yt_channel()
    background_audio = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\onerepublic I aint worried tiktok whistle loop - slowed reverb [mp3].mp3"
    change_on_saturation = "Decrease"
    logger = MyProgressLogger()
   # log(f"YT_channel: {YT_channel}")
    probe = ffmpeg.probe(video_path)
    log(probe)
    format_info = probe.get('format', {})
    bitrate = int(format_info.get('bit_rate', 0))
    video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
    video_codec = video_streams[0]['codec_name'] if video_streams else None
    audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']
    audio_codec = audio_streams[0]['codec_name'] if audio_streams else None

    log(f"subtitle_text: {subtitle_text}")
    log("Before loading YOLO model")



    def group_subtitle_words_in_pairs(subtitle_words):
         chunks = []
         i = 0
         offset = float(subtitle_words[0]['start']) if subtitle_words else 0.0
         while i < len(subtitle_words):
              pair = subtitle_words[i:i+2]

              text_chunk = ''.join([w['word'].strip() + ' ' for w in pair]).strip().upper()

              start = float(pair[0]['start']) - offset

              end = float(pair[-1]['end']) - offset  
              duration = end - start
              start = max(0.0, start)
              duration = max(0.0, duration)
              
              chunks.append({'text': text_chunk, 'start': start, 'duration': duration})

              i += 2
              log(f"[group_subtitle_words_in_pairs] CHUNKS: {chunks}")
         return chunks 
    
    try:
       log(f"[group_subtitle_words_in_pairs] Running now...")
       pairs = group_subtitle_words_in_pairs(subtitle_text)
       log(f"[group_subtitle_words_in_pairs] PAIRS: {pairs}")
    except Exception as e:
         log(f"[group_subtitle_words_in_pairs] Error during grouping of subtitles in pairs. {str(e)} ")
  

    def create_subtitles_from_pairs(pairs):
        from moviepy.video.fx import FadeIn, FadeOut
        text_clips = []
        fade_duration = 0.05

        for c in pairs:
            log(f"text for c in pairs: {c}")
            txt_clip = TextClip(
                text=c['text'],
                font=r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Utils-Video_creation\Fonts\OpenSans-VariableFont_wdth,wght.ttf", 
                font_size=50,
                margin=(10, 10), 
                text_align="center",
                vertical_align="center",
                horizontal_align="center",
                color='white',
                stroke_color="black",
                stroke_width=6,
                size=(1000, 300),
                method="label",
                duration=c['duration']
               ).with_start(c['start']).with_position(('center', 0.60), relative=True)
            txt_clip  = txt_clip.with_effects([FadeIn(fade_duration), FadeOut(fade_duration)])
            text_clips.append(txt_clip)
            log(f"[create_subtitles_from_pairs] Appending: {txt_clip}")
            log(f"Text_clips: {text_clips}")
        return text_clips

        





#------------------------------------------------------------------------#
# CREATES THE (START & END) OF the currentclip from FULL ORIGINAL VIDEO
#------------------------------------------------------------------------#
    full_video = VideoFileClip(video_path)
    clip = full_video.subclipped(start_time, end_time)
    log(f"clip duration: {clip.duration}, clip fps: {clip.fps}, clip width: {clip.w}, clip height: {clip.h}, start_time: {start_time}, end_time: {end_time}, video_path: {video_path}")








#--------------------------------------------------#
# Extrcting frames from original video to a LIST
#--------------------------------------------------#

    log(f"\n\n[Extracting original video frames]  PROCCESS starting...")

    frames = []
    for frame in clip.iter_frames():
        frames.append(frame)
    log(f"[Extracting original video frames] Extracted {len(frames)} frames.\n\n")
    frame_height, frame_width = frame.shape[:2]
    log(f"[CLIP.ITER] Height: {frame_height}, Width: {frame_width}")









#--------------------------------#
# Yolo8/facedetection + Cropping
#--------------------------------#
    log(f"\n\n[detect_and_crop_frames_batch]  PROCCESS starting...")
    cropped_frames = detect_and_crop_frames_batch(frames=frames,batch_size=8)
    log(f"[detect_and_crop_frames_batch] Successfully complete. \n\n")




# ###########################
# ##---------------------###
# ## Enchance & detailed sharpening
# ##---------------------##
# ##########################
#     log(f"\n\n[Enchance & detailed sharpening] PROCCESS starting...")
#     enchanced_frames = []
#     for frame in cropped_frames:
#          frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#          enchanced_frame = enhance_detail_and_sharpness(frame_bgr, clarity_factor=0.4, sharpen_amount=0.4)
#          log(f"[enhance_detail_and_sharpness] appending enchanced frame ...")
#          enchanced_rgb_frame = cv2.cvtColor(enchanced_frame,cv2.COLOR_BGR2RGB)
#          enchanced_frames.append(enchanced_rgb_frame)
#     log(f"[enhance_detail_and_sharpness] Successfully done!\n\n")




# #--------------------------------#
# # GFPGANER & REALESRGAN UPSCALING
# #--------------------------------#
#     log(f"\n\n[GFPGANER & REALESRGAN UPSCALING] PROCCESS starting...")
#     from basicsr.utils.registry import ARCH_REGISTRY
#     ARCH_REGISTRY._obj_map.pop('RRDBNet', None)
#     ARCH_REGISTRY._obj_map.pop('ResNetArcFace', None)
#     from basicsr.archs.rrdbnet_arch import RRDBNet
#     from realesrgan import RealESRGANer 
#     model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,num_block=23, num_grow_ch=32, scale=2)
#     bg_upsampler = RealESRGANer(model_path=r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\gfpgan\weights\RealESRGAN_x2plus.pth", model=model, scale=2)

#     GFPGaner_frames = []
#     from gfpgan import GFPGANer
#     gfpganer = GFPGANer(model_path=r'C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\gfpgan\weights\GFPGANv1.4.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=bg_upsampler)
#     try:
#         log(f"[GFPGANER] starting upscaling frames now....")
#         for frame in tqdm(enchanced_frames, desc="[GFPGAN] Upscaling", unit="frame"):
#             frame_height,frame_width = frame.shape[:2]
#             _, _, gfpganer_enchanced_frame = gfpganer.enhance(frame, has_aligned=False, only_center_face=True, paste_back=True, weight=0.3)
#             enchanced_height,enchanced_width = gfpganer_enchanced_frame.shape[:2]
#             log(f"[GFPGANER] enchanced frame..")
#             log(f"[GFPGANer] Frame input---> height: {frame_height},  width: {frame_width} \n enchanced_frame:  Height: {enchanced_height}, width: {enchanced_width} \n ")
#             GFPGaner_frames.append(gfpganer_enchanced_frame)
#             log(f"[GFPGANER] appending upscaled frame")

#         log(f"Cleared cache and collected garbage")
#         log(f"Gfpganer total frames: {len(GFPGaner_frames)}")
#     except Exception as e:
#          log(f"[GFPGANER] Error during upscaling.. {str(e)}")

#     log(f"[GFPGAN] upscaling finnished....\n\n")





# #----------------------#
# #   FACEENCHANCEMENT
# #----------------------#
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
#          for frame in tqdm(GFPGaner_frames, desc="[FaceEnhancement]  proccessing frames", unit="frame"):
#               log(f"Input frame size: {frame.shape[1]} x {frame.shape[0]}")
#               frame_height,frame_width = frame.shape[:2]
#               frame_bgr = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
#               enchanced_frame, _, _ = Skin_texture_enchancement.process(frame_bgr)
#               log(f"Enhanced frame size: {enchanced_frame.shape[1]} x {enchanced_frame.shape[0]}")
#               enchanced_height,enchanced_width = enchanced_frame.shape[:2]
#               log(f"enchanced_frame height: {enchanced_frame.shape[0]}")
#               FaceEnhancement_frames.append(enchanced_frame)
#               log(f"[FaceEnhancement] Appended enhanced frame")
#               log(f"[FaceEnhancement]Frame input---> height: {frame_height},  width: {frame_width} \n enchanced_frame:  Height: {enchanced_height}, width: {enchanced_width} \n ")
#          log("[FaceEnhancement] Successfully done")
#          torch.cuda.empty_cache()
#          gc.collect()
#          #del GFPGaner_frames

#          log(f"Cleared cache and collected garbage")     
#     except Exception as e:
#             log(f"[FaceEnhancement] Error: {str(e)}")
    


# #-----------------#
# # color/Adjustment
# #-----------------#
#     log(f"\n\n[COLOR ADJUSTMENT] PROCCESS starting...")
#     if change_on_saturation != None and change_saturation == "Increase":
#             FaceEnhancement_frames = [change_saturation(frame,mode=change_on_saturation, amount=0.2) for frame in FaceEnhancement_frames]

#     elif change_on_saturation != None and change_on_saturation == "Decrease":
#             FaceEnhancement_frames = [change_saturation(frame,mode=change_on_saturation, amount=0.2) for frame in FaceEnhancement_frames]







#-------------------------------#
# Creating videoclip from frames
#-------------------------------#
    log(f"\n\n[CREATING VIDEOCLIP] PROCCESS starting...")
    try:
       log(f"[processed_clip] proccessing frames now..")
       processed_clip = ImageSequenceClip(cropped_frames, fps=clip.fps).with_duration(clip.duration)
    except Exception as e:
         log(f"[processed_clip] error during video setup: {str(e)}")
      




#-----------------------------------------------#
# Subtitle creation of text and added on video
#-----------------------------------------------#
    try:
        log(f"Creating subtitles from pairs now..")
        subtitle_clips = create_subtitles_from_pairs(pairs)
    except Exception as e:
            log(f"error during  creation of subtitleclips: {str(e)}")




#-----------------------------------------------#
#     Adds Logo/overlay for YT_channel
#-----------------------------------------------#
    YT_channel = None

    try:
        global Upload_YT_count
        Upload_YT_count += 1
        if Upload_YT_count == 1:
             YT_channel = "LR_Youtube"
        elif Upload_YT_count == 2:
             YT_channel = "LRS_Youtube"
        elif Upload_YT_count == 3:
             YT_channel = "MR_Youtube"
        elif Upload_YT_count == 4:
             YT_channel = "LM_Youtube"

        if YT_channel == "LR_Youtube":
                overlay_clip = VideoFileClip(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\alpha_mask\YT_logo\LOGO_LR.mp4",has_mask=True)
        
        elif YT_channel == "LRS_Youtube":
                 overlay_clip = VideoFileClip(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\alpha_mask\YT_logo\LOGO_LRS.mp4",has_mask=True)

        elif YT_channel == "MR_Youtube":
            overlay_clip = VideoFileClip(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\alpha_mask\YT_logo\LOGO_MR.mp4",has_mask=True)

        elif YT_channel == "LM_Youtube":
             overlay_clip = VideoFileClip(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\alpha_mask\YT_logo\LOGO_LM.mp4",has_mask=True)
        else:
             raise ValueError(f"Error No {YT_channel} exists.")
        
        if Upload_YT_count == 4:
             Upload_YT_count = 0

        overlay_clip = overlay_clip.subclipped(0, clip.duration)
        logo_mask = overlay_clip.to_mask()
        logo_with_mask = overlay_clip.with_mask(logo_mask) 





#-----------------------------------------------#
# Adds Subtitles + Logo to the video
#-----------------------------------------------#
        log(f"Adding subtitle to the video....")
        final_clip = CompositeVideoClip(
                    [processed_clip.with_position('center')] + subtitle_clips + [logo_with_mask.with_position('center','bottom')],
                    size=processed_clip.size
                )
        
    except Exception as e:
         log(f"Error during: finalizing clip with subtitle_clips:  {str(e)}")


#-------------------------------#
# Adds Background Music to video
#-------------------------------#
    if background_audio != None:
        background_music_path = background_audio
        final_clip.audio = mix_audio(clip.audio, background_music_path, bg_music_volume=0.15)
        log(f"adding audio to video...")
    else:
        final_clip.audio = clip.audio
        log(f"keeping original audio")





#-----------------------------------------------------#
#    Adds fade in/out to the video and sets the FPS
#------------------------------------------------------#
    final_clip = FadeIn(duration=0.1).apply(final_clip)
    final_clip = FadeOut(duration=0.1).apply(final_clip)
    final_clip.fps = clip.fps
    log(f"Final clip fps: {final_clip.fps}")


#-----------------------------------------------------#
#    Cleans cpu/gpu memory & Frames
#------------------------------------------------------#
    del frames
    #del cropped_frames
    clean_get_gpu_memory()
    #del enchanced_frames



#-----------------------------------------------------#
#    Writes the final Video
#------------------------------------------------------#
    output_dir = "./Video_clips"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{video_name}.mp4")
    final_clip.write_videofile(
        out_path,
        logger=logger,
        codec="h264_nvenc",
        preset="p7",
        audio_codec=audio_codec or "aac",
        threads=6,
        fps=final_clip.fps,
        ffmpeg_params=[
            "-crf", "5",
            "-rc:v", "vbr_hq",
            "-vf", "eq=brightness=0.08",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-b:a", "192k",  
        ],

        remove_temp=True
    )
    full_video.close()
    clip.close()




#-----------------------------------------------------#
#    Boosts x2 FPS on full video
#------------------------------------------------------#
    from RIFE_FPS import run_rife
    try:
      output_video = run_rife(out_path)
      log(f"Interpolated video saved to: {output_video}")
      log(f"done with interpolation")
    except Exception as e:
         log(f"error during frame interpolation: {str(e)}")
    finally:
        os.remove(out_path)
    log(f"video is completed: output path : {out_path}, video name: {video_name} video_fps: {clip.fps}, codec: {video_codec}, bitrate: {bitrate}, audio_codec: {audio_codec}, subtitles: {subtitle_text}")
    log(f"Final video resolution (width x height): {final_clip.size[0]} x {final_clip.size[1]}") 






#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# A Agent creates optimized (Title, Description, Hashtags, Tags, category, publishAt) after analyzing similar trending videos related to input video &  Uploads the video to Youtube
# - Reloads the Global Model
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    try:
        from Agent_AutoUpload.Upload_youtube import upload_video
        log(f"\n\n\n\n\n\n\n\n\n-------------------------------------------------------------------\n\n\n\n")
        log("Youtube uploading & agent STARTING....")
        try:
             global Global_model
             del Global_model    
        except NameError:
                pass
        
        clean_get_gpu_memory()
        Global_model = Reload_and_change_model(model="Qwen7b")
        social_media = upload_video(model=Global_model,file_path=output_video,YT_channel=YT_channel)
        log(f"Done with uploading to {social_media}")
    except Exception as e:
         log(f"error during uploading: {str(e)}")

    finally:
         del Global_model
         clean_get_gpu_memory()
         Reload_and_change_model("phi-4")
 



#-----------------------------------------------------#
# Verifies subtitles & video start time/video end time
#-----------------------------------------------------#

def run_video_short_creation_thread(video_url,start_time,end_time,subtitle_text):
        global Video_count
        Video_count += 1
        current_count = Video_count


        def truncate_audio(audio_path, start_time, end_time, output_path):
            """
            Truncate an audio file from start_time to end_time and save it to output_path.

            start_time, end_time: in seconds (float or int)
            """
            audio = AudioSegment.from_file(audio_path)
            
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            log(f"start_ms: {start_ms}, end_ms: {end_ms}")

            truncated = audio[start_ms:end_ms]
            truncated.export(output_path, format="wav") 
            return output_path
        
        try:
            log(f"RUNNING --> [run_video_short_creation_thead]: video_url: {video_url}, start_time: {start_time}, end_time: {end_time}")
            text_video_path = video_url
            video_start_time = start_time
            video_end_time = end_time

      
            device = "cuda"
            tool = SpeechToText_short_creation_thread(device=device)
            tool.setup()

            audio_path = Global_state.get_current_audio_path()
            with tempfile.TemporaryDirectory() as temp_dir:
                log(f"Temp dir created:{temp_dir}")
                truncated_audio_path = os.path.join(temp_dir, f"temp_audio_{current_count}.wav")
                log(f"Truncated audio clip: {truncated_audio_path}")
                audio_for_clip = truncate_audio(audio_path, video_start_time, video_end_time, truncated_audio_path)

                subtitle_text = re.sub(r"\[\d+\.\d+s\s*-\s*\d+\.\d+s\]", "", subtitle_text)
                subtitle_text = re.sub(r"\s+", " ",subtitle_text).strip()
                log(f"subtitletext cleaned: {subtitle_text}")
        
                result = tool.forward({"audio": audio_for_clip,"subtitle_text": subtitle_text, "original_start_time": video_start_time, "original_end_time:": video_end_time})

                crafted_Subtitle_text = result["matched_words"]
                new_video_start_time = float(result["video_start_time"])
                new_video_end_time = float(result["video_end_time"])
                log(f"[run_video_short_creation_thread] creating video now... \n start_time: {new_video_start_time} \n end_time: {new_video_end_time},  \n subtitle_text: {crafted_Subtitle_text}")

                



                text_video_title = "short1" + str(current_count)
                create_short_video(video_path=text_video_path, start_time=new_video_start_time, end_time = new_video_end_time, video_name = text_video_title, subtitle_text=crafted_Subtitle_text)

        except Exception as e:
                log(f"[ERROR] during execution: {str(e)}")








#---------------------------------------------------------------------------#
# Retrieves Video Creation tasks from a Queue & runs creation of that video
#---------------------------------------------------------------------------#
def video_creation_worker():
     global ActiveProgress
     while True:
          try:
             video_url,  final_start_time, final_end_time, subtitle_text = Global_state.video_task_que.get()
             log(f"Retrieved WORK: {video_url}\n {final_start_time}\n {final_end_time}\n {subtitle_text}\n")
          except queue.Empty:
                log(f"[video_creation_worker] Que empty!")
                break 
          try:
             log(f"\n[video_creation_worker] Current work being proccessed...[ video_url: {video_url}, start_time: {final_start_time}, end_time: {final_end_time}, text: {subtitle_text} to que]")
             log(f"[video_creation_worker] Processing video task: {video_url}, {final_start_time}-{final_end_time}")
             ActiveProgress = True
             run_video_short_creation_thread(video_url, final_start_time, final_end_time, subtitle_text)
             log(f"Done Creating Video \n")
             ActiveProgress=False
          except Exception as e:
               log(f"[video_creation_worker] error during video_creation_worker: {str(e)}")
               raise ValueError(f"[video_creation_worker]Error during video creation")
          finally:
               log("Video Successfully Done!")
               Global_state.video_task_que.task_done()
    



#-------------------------------------------------------------------------------------------#
# Listens & waits for a queue to be empty before procceeding with Transcript Reasoning Agent
#-------------------------------------------------------------------------------------------#
def wait_for_proccessed_video_complete(queue: Queue, check_interval=30):
    global ActiveProgress
    """Blocks until the queue is empty, checking every `check_interval` seconds."""
    log(f"\n\n\n\n\n\n[wait_for_proccessed_video_complete]")
    while not queue.empty() and ActiveProgress:
          log(f"[wait_for_proccessed_video_complete]  waiting for video_task_que to be empty: items remaining: {queue.qsize()}")
          time.sleep(check_interval)
    log("[wait_for_proccessed_video_complete]✅ video_task_que is now empty.")


        




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
    Global_state.set_current_textfile(agent_saving_path)
    log(f"agent_saving_path: {agent_saving_path}")
    global Global_model

    with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Prompt_templates\verify_agent_system_prompt.yaml", "r", encoding="utf-8") as f:
         verify_system_prompt = yaml.safe_load(f)

    def save_thought_and_code(step_output):
            text = getattr(step_output, "model_output", "") or ""
            
            thought = ""
            code = ""
            if "Thought:" in text and "Code:" in text:
                thought = text.split("Thought:")[1].split("Code:")[0].strip()
                code = text.split("Code:")[1].strip()
            else:
                code = text.strip()
            reasoning_log.append(f"Thought:\n{thought}\n\nCode:\n{code}\n\n")
  
    final_answer = FinalAnswerTool()
    create_motivational_short_agent = CodeAgent(
        model=Global_model,
        tools=[create_motivationalshort,Delete_rejected_line,final_answer],
        max_steps=1,
        prompt_templates=verify_system_prompt,
        stream_outputs=True,
        verbosity_level=4,
    )
    reasoning_log = []
    create_motivational_short_agent.step_callbacks.append(save_thought_and_code)


    
    with open(agent_saving_path, "r", encoding="utf-8") as f:
             saved_quotes_text = f.read()
             if not saved_quotes_text.strip():
                  log("empty, break")
                  return
    
    Blocks = re.findall(r"===START_TEXT===.*?===END_TEXT===", saved_quotes_text, re.DOTALL)

    chunk_size = 2
    chunks = [Blocks[i:i+chunk_size] for i in range(0, len(Blocks), chunk_size)]

    for idx, chunk in enumerate(chunks,1):
         combined_text = "\n".join(chunk)
         print(f"CHUNK PRINT: {combined_text}")

         task = f"""
                Analyze all the textblock's, reject those textblocks that does not qualify as a standalone Quote/Advice by deleting them using `Delete_rejected_line` tool that is not valid for a motivational short, and run `create_motivationalshort` tool for those that are valid.  
                now start chain of thought reasoning over textblock/textblock's
                here is the text to analyze:
                    [{combined_text}] 
                    """
        
         model_response = create_motivational_short_agent.run(task=task)
         save_full_io_to_file(modelname="VerifyAgent",input_chunk=combined_text,reasoning_steps=reasoning_log, model_response=model_response,  file_path=r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\debug_performance\VerifyAgentRun_data.txt")
    
    with open(agent_saving_path, "w", encoding="utf-8") as f:
         f.write("")

    log("All chunks processed. File has been emptied.")
    del create_motivational_short_agent




#---------------------------------------------------------------------------------------------------------------------------------#
#   Agent that analyzes text  from transcript by reading it (chunk for chunk) --->  (saves Quote identified in podcast transcript.
#---------------------------------------------------------------------------------------------------------------------------------#
def Transcript_Reasoning_AGENT(transcripts_path,agent_txt_saving_path):
    from Custom_Agent_Tools import ChunkLimiterTool
    log(f"✅ Entered Transcript_Reasoning_AGENT() transcript_path: {transcripts_path}, agent_txt_saving_path: {agent_txt_saving_path}")
  
    global Global_model
    loaded_reasoning_agent_prompts = r'C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Prompt_templates\System_prompt_TranscriptReasoning.yaml'
    with open(loaded_reasoning_agent_prompts, 'r', encoding='utf-8') as f:
            Prompt_template = yaml.safe_load(f)

    final_answer = FinalAnswerTool()
    Reasoning_Text_Agent = CodeAgent(
        model=Global_model,
        tools=[SaveMotivationalText,final_answer],
        max_steps=2,
        prompt_templates=Prompt_template,
    )
    chunk_limiter = ChunkLimiterTool()

        
    log(f"transcript_path that is being proccessed inside func[Transcript_Reasoning_Agent]: {transcripts_path}")
    transcript_title = os.path.basename(transcripts_path)
    log(f"transcript title: {transcript_title}")
    log(f"\nProcessing new transcript: {transcripts_path}")

        
    def save_thought_and_code(step_output):
            text = getattr(step_output, "model_output", "") or ""
            
            thought = ""
            code = ""
            if "Thought:" in text and "Code:" in text:
                thought = text.split("Thought:")[1].split("Code:")[0].strip()
                code = text.split("Code:")[1].strip()
            else:
                code = text.strip()
            reasoning_log.append(f"Thought:\n{thought}\n\nCode:\n{code}\n\n")

   
    reasoning_log = []
    Reasoning_Text_Agent.step_callbacks.append(save_thought_and_code)
    chunk_limiter.reset()
    while True:
        reasoning_log.clear()
        try:
            log(f"transcript_path for chunk tool : {transcripts_path}")
            chunk = chunk_limiter.forward(file_path=transcripts_path, max_chars=5000)
        except Exception as e:
                log(f"Error during chunking from file {transcripts_path}: {e}")
                break



        if not chunk.strip():
                log("Finished processing current transcript. Now exiting func [Transcript Reasoning Agent]")
                del Reasoning_Text_Agent
                verify_saved_text_agent(agent_txt_saving_path)
                wait_for_proccessed_video_complete(Global_state.video_task_que)
                break
        


        task = f"""
                Here is the chunk you will analyze:
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
        save_full_io_to_file(
            modelname="Reasoning_Agent",
            input_chunk=chunk,
            reasoning_steps=reasoning_log,
            model_response=result,
            file_path=r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\debug_performance\VerifyAgentRun_data.txt"
        )
        chunk_limiter.called = False 



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
        log(f"Transcript already exists: {txt_output_path}, audio exists: {audio_path}")
        Global_state.transcript_queue.put((video_path, txt_output_path,agent_text_saving_path))
        Global_state.set_current_audio_path(audio_path)
        log(f"Global audio_path: {audio_path}")
        log(f"Enqueued existing transcript for GPU processing: {txt_output_path}")
        return
    
    try:
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-af", "afftdn=nr=20:nf=-30:tn=1"
            "-acodec", "pcm_s16le",
            audio_path
        ]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        Global_state.set_current_audio_path(audio_path)
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

        Global_state.transcript_queue.put((video_path, txt_output_path, agent_text_saving_path))
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
    log("GPU worker started")
    torch.backends.cudnn.benchmark = True
    global Global_model
    Global_model = TransformersModel(
            model_id = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\microsoft\unsloth\phi-4-mini-instruct-FinedTuned_version2",
            load_in_4bit=True,
            device_map="auto",
            torch_dtype="auto",
            use_flash_attn=True,
            max_new_tokens=30000,
            do_sample=False,
     )


    log(f"Loaded Global_model on device")
    itemcount = 0

    while True:
        item = Global_state.transcript_queue.get()

        log(f"item: {item}")
        if item is None and itemcount <= 0:
            log("Shutdown signal received, exiting GPU worker")
            Global_state.transcript_queue.task_done()
            break

        itemcount += 1

        video_path_url, transcript_text_path,agent_txt_saving_path = item
        Global_state.set_current_videourl(video_path_url)
        log(f"Processing {video_path_url} & {transcript_text_path} in GPU worker, agent is using txt path to save: {agent_txt_saving_path}")

        log(f"GPU lock acquired by gpu_worker for {video_path_url}")

        try:
            Transcript_Reasoning_AGENT(transcript_text_path, agent_txt_saving_path)
            log(f"Transcript_Reasoning_AGENT has exited...")
        finally:
            log(f"GPU lock released by gpu_worker for {video_path_url}")
            Global_state.transcript_queue.task_done()





if __name__ == "__main__":
    clean_get_gpu_memory()
    Clean_log_onRun()

    worker_thread = threading.Thread(target=video_creation_worker,name="Video_creation(THREAD)")
    worker_thread.start()


    video_paths = [
        r"c:\Users\didri\Documents\The Best Moments Of Modern Wisdom (2024).webm",

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




    gpu_thread = threading.Thread(target=gpu_worker, name="Agent_Run(THREAD)")
    gpu_thread.start()
    Global_state.transcript_queue.join() 
    gpu_thread.join()
    worker_thread.join()


