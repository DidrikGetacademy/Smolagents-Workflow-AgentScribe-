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
from smolagents import TransformersModel, FinalAnswerTool, SpeechToTextTool, CodeAgent, tool,SpeechToTextToolCPU
from Agents_tools import ChunkLimiterTool
import gc
import yaml

import torchvision.transforms.functional as F
sys.modules['torchvision.transforms.functional_tensor'] = F
import subprocess
from smolagents import SpeechToTextTool
from moviepy import VideoFileClip, ImageSequenceClip, TextClip, CompositeVideoClip,vfx,AudioFileClip,afx,CompositeAudioClip
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
from moviepy.video.fx.FadeIn import FadeIn
from moviepy.video.fx.FadeOut import FadeOut
import threading
import queue
import torch
import datetime
import re 
from queue import Queue
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(f"Total Used: {mem_info.used/1e9:.1f}GB")
Chunk_saving_text_file = r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Logging_and_filepaths\saved_transcript_storage.txt"
Final_saving_text_file=r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Logging_and_filepaths\final_saving_motivational.txt"
model_path_SwinIR_color_denoise15_pth = r"c:\Users\didri\Desktop\LLM-models\Video-upscale-models\SwinIR-M_noise15.pth"
model_path_SwinIR_color_denoise15_onnx = r"c:\Users\didri\Desktop\LLM-models\Video-upscale-models\SwinIR-M_noise15.onnx"
model_path_Swin_BSRGAN_X4_pth = r"c:\Users\didri\Desktop\LLM-models\Video-upscale-models\BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
model_path_Swin_BSRGAN_X4_onnx = r"c:\Users\didri\Desktop\LLM-models\Video-upscale-models\BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.onnx"
model_path_realesgran_x2_pth = r"c:\Users\didri\Desktop\LLM-models\Video-upscale-models\RealESRGAN_x2plus.pth"
video_task_que = queue.Queue()
gpu_lock = threading.Lock()
transcript_queue = queue.Queue()
count_lock = threading.Lock()

_current_video_url: str = None
def set_current_videourl(url: str):
    global _current_video_url
    _current_video_url = url

def get_current_videourl() -> str:
    global _current_video_url
    return _current_video_url



_current_audio_url: str = None
def set_current_audio_path(url: str):
     global _current_audio_url
     _current_audio_url = url
     log(f"[set_current_audio_path]: {_current_audio_url}")

def get_current_audio_path() -> str:
     global _current_audio_url
     return _current_audio_url


def set_current_textfile(url: str):
    global _current_agent_saving_file
    _current_agent_saving_file = url

def get_current_textfile() -> str:
     global _current_agent_saving_file 
     return _current_agent_saving_file


#- en idee er at man har en ekstra agent som kan gå igjennom alle lagde videoclips til slutt og ser om det går ann og lage noe montage, en shorts video som innholder motivational quotes/advices fra videoklips (resultat blir da at agenten  velger rekkefølge  på videoen som skal slå sammen til 1. video, med tanke at (det skal være motiverende og det må passe sammen)
def create_motivational_montage_agent(clips: List[str], output_path: str):
    return 

def  clear_queue(q: Queue):
     with q.mutex:
          q.queue.clear()
          q.all_tasks_done.notify_all()
          q.unfinished_tasks = 0


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
          
        #cv2.imwrite("before_gfpgan.jpg", cv2.cvtColor(cropped_frames[0], cv2.COLOR_RGB2BGR))
        #cv2.imwrite("after_gfpgan.jpg", cv2.cvtColor(restored_frames[0], cv2.COLOR_RGB2BGR))
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

    # === 1) Detail Layer (Clarity) ===
    # Bruk bilateral filter for å lage en glatt versjon
    smooth = cv2.bilateralFilter(frame_bgr, d=9, sigmaColor=75, sigmaSpace=75)

    # Trekker smooth ut av original for å få detaljene
    detail_layer = cv2.subtract(frame_bgr, smooth)

    # Booster detaljene
    boosted_detail = cv2.addWeighted(detail_layer, clarity_factor, detail_layer, 0, 0)

    # Legger detaljene tilbake til originalen
    clarity_frame = cv2.add(frame_bgr, boosted_detail)

    # === 2) Mild sharpen ===
    # Unsharp mask: original + (original - blur)
    blur = cv2.GaussianBlur(clarity_frame, (0, 0), 3)
    sharpened = cv2.addWeighted(clarity_frame, 1.0 + sharpen_amount, blur, -sharpen_amount, 0)

    return sharpened

def sharpen_frame_naturally(frame_bgr):
            from PIL import ImageFilter,Image
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            sharpned_pil = pil_img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=113,threshold=2))
            sharpned_rgb = np.array(sharpned_pil)
            sharpened_bgr = cv2.cvtColor(sharpned_rgb, cv2.COLOR_RGB2BGR)
            print(f"sharpening completed")
            return sharpened_bgr


def downscale_to_size( img: np.ndarray, width: int, height: int) -> np.ndarray:
            """
            Downscale an image to a specific width and height using Lanczos interpolation.
            """
            new_size = (width, height)
            print(f"downscale complete on img: {img},  {new_size}")
            return cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)



def change_video_resolution(video_path, target_width, target_height, output_path=None):
        if output_path is None:
            output_path = "./converted_original_for_quality_test.mp4"
        (
            ffmpeg
            .input(video_path)
            .output(output_path, vf=f'scale={target_width}:{target_height}', preset='slow', crf=18)
            .overwrite_output()
            .run()
        )
        return output_path
    


def get_video_resolution(video_path):
        cmd = [
            "ffprove",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        width, height = map(int, result.stdout.strip().split('x'))
        return width,height    


def test_videoquality_comparison(original_video_path, new_video_path):
        import subprocess
        try: 
            orig_width, orig_height = get_video_resolution(original_video_path)
            new_width, new_height = get_video_resolution(new_video_path)
            if (orig_width,orig_height) != (new_width, new_height):
               print(f"video sizes differ: original {orig_width}x{orig_height}, new {new_width}x{new_height}")
            

            print(f"both videos have the same resolution continuing...: {orig_width}x {orig_height}")

            if orig_width == 1920 and orig_height == 1080:
                model_path = "./vmaf_float_v0.6.1.json"
            elif orig_width == 3840 and orig_height == 2160:
                model_path = "./vmaf_float_4k_v0.6.1.json"

            else: 
                print("resolution not supported")
                return
            
            cmd = [
                "ffmpeg",
                "-i", new_video_path,
                "-i", original_video_path,
                "-lavfi", f"libvmaf={model_path}:log_fmt=json:log_path./vmaf_output.json",
                "-f", "null",
                "-"
            ]
            subprocess.run(cmd)
        except Exception as e:
            print(f"error during testing of videoquality!!!! reason: {str(e)}")





from proglog import ProgressBarLogger

class MyProgressLogger(ProgressBarLogger):
    def callback(self, **changes):
        for param, value in changes.items():
            print(f"{param}: {value}")
            log(f"{param}: {value}")
            

 
def create_short_video(video_path, start_time, end_time, video_name, subtitle_text):
    background_audio = None
    change_on_saturation = "Increase"
    logger = MyProgressLogger()
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
         while i < len(subtitle_words):
              pair = subtitle_words[i:i+2]

              text_chunk = ''.join([w['word'].strip() + ' ' for w in pair]).strip().upper()

              start = float(pair[0]['start'])

              end = float(pair[-1]['end'])
              duration = end - start
              
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
        text_clips = []
        for c in pairs:
            txt_clip = TextClip(
                text=c['text'],
                font=r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Utils-Video_creation\Fonts\OpenSans-VariableFont_wdth,wght.ttf", 
                font_size=45,
                margin=(10, 10), 
                text_align="center",
                vertical_align="center",
                horizontal_align="center",
                color='white',
                stroke_color="black",
                stroke_width=3,
                size=(1000, None),
                method="label",
                duration=c['duration']
            ).with_position(('center', 0.60), relative=True
            ).with_start(c['start'])
            text_clips.append(txt_clip)
            log(f"[create_subtitles_from_pairs] Appending: {txt_clip}")
        return text_clips

        


    
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
                print(f"batch: {i} - {i + batch_size}")
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
    



#########################################################################
##--------------------------------------------------------------------###
## CREATES THE (START & END) OF the currentclip from FULL ORIGINAL VIDEO
##--------------------------------------------------------------------###
#########################################################################

    full_video = VideoFileClip(video_path)
    clip = full_video.subclipped(start_time, end_time)
    log(f"clip duration: {clip.duration}, clip fps: {clip.fps}, clip width: {clip.w}, clip height: {clip.h}, start_time: {start_time}, end_time: {end_time}, video_path: {video_path}")



##############################################################
##----------------------------------------------------------###
## Extrcting frames from original video to a LIST
##----------------------------------------------------------###
##############################################################
    log(f"\n\n[Extracting original video frames]  PROCCESS starting...")

    frames = []
    for frame in clip.iter_frames():
        frames.append(frame)
    log(f"[Extracting original video frames] Extracted {len(frames)} frames.\n\n")
    frame_height, frame_width = frame.shape[:2]
    log(f"[CLIP.ITER] Height: {frame_height}, Width: {frame_width}")





###############################
##--------------------------###
## Yolo8/facedetection + Cropping
##--------------------------###
###############################
    log(f"\n\n[detect_and_crop_frames_batch]  PROCCESS starting...")
    cropped_frames = detect_and_crop_frames_batch(frames=frames,batch_size=8)
    log(f"[detect_and_crop_frames_batch] Successfully complete. \n\n")







###########################
##---------------------###
## Enchance & detailed sharpening
##---------------------##
##########################
    log(f"\n\n[Enchance & detailed sharpening] PROCCESS starting...")
    enchanced_frames = []
    for frame in cropped_frames:
         frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
         enchanced_frame = enhance_detail_and_sharpness(frame, clarity_factor=0.4, sharpen_amount=0.4)
         log(f"[enhance_detail_and_sharpness] appending enchanced frame ...")
         enchanced_rgb_frame = cv2.cvtColor(enchanced_frame,cv2.COLOR_BGR2RGB)
         enchanced_frames.append(enchanced_rgb_frame)

    log(f"[enhance_detail_and_sharpness] Successfully done!\n\n")










###########################
##---------------------###
#GFPGANER & REALESRGAN UPSCALING
##---------------------##
##########################
    log(f"\n\n[GFPGANER & REALESRGAN UPSCALING] PROCCESS starting...")
    from basicsr.utils.registry import ARCH_REGISTRY
    ARCH_REGISTRY._obj_map.pop('RRDBNet', None)
    ARCH_REGISTRY._obj_map.pop('ResNetArcFace', None)
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer 
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,num_block=23, num_grow_ch=32, scale=2)
    bg_upsampler = RealESRGANer(model_path=r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\gfpgan\weights\RealESRGAN_x2plus.pth", model=model, scale=2)

    GFPGaner_frames = []
    from gfpgan import GFPGANer
    gfpganer = GFPGANer(model_path=r'C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\gfpgan\weights\GFPGANv1.4.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=bg_upsampler)
    try:
        log(f"[GFPGANER] starting upscaling frames now....")
        for frame in tqdm(enchanced_frames, desc="[GFPGAN] Upscaling", unit="frame"):
            frame_height,frame_width = frame.shape[:2]
            _, _, gfpganer_enchanced_frame = gfpganer.enhance(frame, has_aligned=False, only_center_face=True, weight=0.3)
            enchanced_height,enchanced_width = gfpganer_enchanced_frame.shape[:2]
            log(f"[GFPGANER] enchanced frame..")
            log(f"[GFPGANer] Frame input---> height: {frame_height},  width: {frame_width} \n enchanced_frame:  Height: {enchanced_height}, width: {enchanced_width} \n ")
            GFPGaner_frames.append(gfpganer_enchanced_frame)
            log(f"[GFPGANER] appending upscaled frame")

        log(f"Cleared cache and collected garbage")
    except Exception as e:
         log(f"[GFPGANER] Error during upscaling.. {str(e)}")

    log(f"[GFPGAN] upscaling finnished....\n\n")









###########################
##---------------------###
#FACEENCHANCEMENT
##---------------------##
##########################
    log(f"\n\n[FACEENCHACEMENT] PROCCESS starting...")

    
    class Args:
        model = 'GPEN-BFR-512'
        task = 'FaceEnhancement'
        key = None
        in_size = 512
        out_size = None
        channel_multiplier = 2
        narrow = 1
        alpha = 1
        use_sr = True
        use_cuda = True
        save_face = False
        aligned = False
        sr_model = 'realesrnet'
        sr_scale = 1
        tile_size = 0
        ext = '.jpg'


    Skin_texture_enchancement = FaceEnhancement(
         Args,
        in_size=Args.in_size,
        model=Args.model,
        use_sr=Args.use_sr,
        device='cuda' if Args.use_cuda else 'cpu'
    )

    FaceEnhancement_frames = []
    try:
         for frame in tqdm(GFPGaner_frames, desc="[FaceEnhancement]  proccessing frames", unit="frame"):
              frame_height,frame_width = frame.shape[:2]
              frame_bgr = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
              enchanced_frame, _, _ = Skin_texture_enchancement.process(frame_bgr)
              enchanced_height,enchanced_width = enchanced_frame.shape[:2]
              log(f"enchanced_frame height: {enchanced_frame.shape[0]}")
              FaceEnhancement_frames.append(enchanced_frame)
              log(f"[FaceEnhancement] Appended enhanced frame")
              log(f"[FaceEnhancement]Frame input---> height: {frame_height},  width: {frame_width} \n enchanced_frame:  Height: {enchanced_height}, width: {enchanced_width} \n ")
         log("[FaceEnhancement] Successfully done")
         torch.cuda.empty_cache()
         gc.collect()
         del GFPGaner_frames

         log(f"Cleared cache and collected garbage")     
    except Exception as e:
            log(f"[FaceEnhancement] Error: {str(e)}")
    



#####################
##--------------.--##
# color/ADJUSTMENT
##-----------------##
#####################
    log(f"\n\n[COLOR ADJUSTMENT] PROCCESS starting...")
    if change_on_saturation != None and change_saturation == "Increase":
            FaceEnhancement_frames = [change_saturation(frame,mode=change_on_saturation, amount=0.2) for frame in FaceEnhancement_frames]

    elif change_on_saturation != None and change_on_saturation == "Decrease":
            FaceEnhancement_frames = [change_saturation(frame,mode=change_on_saturation, amount=0.2) for frame in FaceEnhancement_frames]


##############################
##--------------------------##
# MAKING videoclip from frames
##--------------------------##
##############################

    log(f"\n\n[CREATING VIDEOCLIP] PROCCESS starting...")
    try:
       log(f"[processed_clip] proccessing frames now..")
       processed_clip = ImageSequenceClip(FaceEnhancement_frames, fps=clip.fps).with_duration(clip.duration)
    except Exception as e:
         log(f"[processed_clip] error during video setup: {str(e)}")
      



################################################
##-------------------------------------------###
## Subtitle creation of text and added on video
##-------------------------------------------###
################################################

    try:
        log(f"Creating subtitles from pairs now..")
        subtitle_clips = create_subtitles_from_pairs(pairs)
    except Exception as e:
            log(f"error during  creation of subtitleclips: {str(e)}")


    try:
        log(f"Adding subtitle to the video....")
        final_clip = CompositeVideoClip(
                    [processed_clip.with_position('center')] + subtitle_clips,
                    size=processed_clip.size
                )
        
    except Exception as e:
         log(f"Error during: finalizing clip with subtitle_clips:  {str(e)}")

###############################
##--------------------------###
## Audio creation
##--------------------------###
###############################
    def mix_audio(original_audio, background_music_path, bg_music_volume=0.15):
        bg_music = AudioFileClip(background_music_path)

        if bg_music.duration < original_audio.duration:
            bg_music = afx.audio_loop(bg_music, duration=original_audio.duration)
        else:
            bg_music = bg_music.subclip(0, original_audio.duration)
        
        bg_music = bg_music.volumex(bg_music_volume)
        original_audio = original_audio.volumex(1.0)

        mixed_audio = CompositeAudioClip([original_audio, bg_music])

        return mixed_audio
        
    if background_audio != None:
        background_music_path = r"c:\Users\didri\AppData\Local\CapCut\Videos\Video Tools\audio\Documentary Cinematic Violin by Infraction [No Copyright Music] # Life Goes On [mp3].mp3"
        final_clip.audio = mix_audio(clip.audio, background_music_path, bg_music_volume=0.15)
        log(f"adding audio to video...")
    else:
        final_clip.audio = clip.audio
        log(f"keeping original audio")


##################################################
##----------------------------------------------###
## Final ADJUSTMENT before finalizing clip
##----------------------------------------------###
###################################################
    final_clip = FadeIn(duration=0.1).apply(final_clip)
    final_clip = FadeOut(duration=0.1).apply(final_clip)

    log(f"video original fps: {clip.fps}")
    log(f"video now fps: {final_clip.fps}")
    output_dir = "./Video_clips"

    del frames
    del cropped_frames
    del enchanced_frames
    torch.cuda.empty_cache()
    gc.collect()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{video_name}.mp4")
    final_clip.write_videofile(
        out_path,
        logger=logger,
        codec="h264_nvenc",
        preset="p7",
        audio_codec=audio_codec or "aac",
        bitrate=str(bitrate) or "4000k",
        threads=8,
        fps=clip.fps,
        ffmpeg_params=[
             "-crf", "18",
             "-vf", "format=yuv420p"],
        remove_temp=True
    )
    
    log(f"video is completed: output path : {out_path}, video name: {video_name} video_fps: {clip.fps}, codec: {video_codec}, bitrate: {bitrate}, audio_codec: {audio_codec}, subtitles: {subtitle_text}")
    log(f"Final video resolution (width x height): {final_clip.size[0]} x {final_clip.size[1]}")  
    full_video.close()
    clip.close()
    clear_queue(video_task_que)
    torch.cuda.empty_cache()
    gc.collect()
    log(f"Cleared cache and collected garbage")





global count
count = 0
def run_video_short_creation_thread(video_url,start_time,end_time,subtitle_text):
        global count
        count += 1
        current_count = count
        try:
            log(f"RUNNING --> [run_video_short_creation_thead]: video_url: {video_url}, start_time: {start_time}, end_time: {end_time}")
            text_video_path = video_url
            video_start_time = start_time
            video_end_time = end_time
            text_video_title = "short1" + str(current_count)
            try:

               log(f"Subtitles: {subtitle_text}")
   
               log(f"Subtitle passed to the [create_short_video] --> subtitle_text_tuple: {subtitle_text}")
    
               try:
                  log(f"[run_video_short_creation_thread] creating video now... \n start_time: {start_time} \n end_time: {end_time}, \n video_name: {text_video_title}, \n subtitle_text: {subtitle_text}")
                  create_short_video(video_path=text_video_path, start_time=video_start_time, end_time = video_end_time, video_name = text_video_title,subtitle_text=subtitle_text)
               except Exception as e:
                  log(f"[run_video_short_creation_thread] error during creation of video : {str(e)}")
               text_video_path = ""
               video_start_time = None
               video_end_time = None
               text_video_title = ""
            except Exception as e:
                log(f"[run_video_short_creation_thread] error during [create_short_video] {str(e)}")
        except Exception as e:
          import traceback
          log("[run_video_short_creation_thread][ERROR] in run_video_short_creation_thread:")
          traceback.print_exc()






def parse_multiline_block(block_text):
    """
    1) Splitter blokken i linjer.
    2) Finner ALLE '[start - end] tekst' — uansett hvordan de står.
    3) Bygger NY tekst med én linje per subtitle.
    4) Returnerer start_time (første), end_time (siste) OG den nye teksten.
    """
    import re
    log(f"block_text: {block_text}")

    pattern = re.compile(r"\[(\d+\.\d+)s\s*-\s*(\d+\.\d+)s\]\s*([^\[]+)")
    matches = pattern.findall(block_text)

    if not matches:
         return None, None, ""
    new_text = "\n".join(
         f"[{start}s - {end}s] {text.strip()}"
         for start, end, text in matches 
         
    )

    Video_start_time = float(matches[0][0])
    Video_end_time = float(matches[-1][1])

    log(f"[parse_multiline_block] start_time: {Video_start_time}")
    log(f"[parse_multiline_block] end_time: {Video_end_time}")
    log(f"[parse_multiline_block] new_text: {new_text}")

    return Video_start_time, Video_end_time, new_text


@tool
def SaveMotivationalText(text: str, text_file: str) -> None:
    """Save motivational text for motivational shorts video, the text that meets task criteria  to a file with a timestamp.
    Args:
         text: The text to be saved. To avoid syntax errors, wrap the string in triple quotes 
              when calling this function, especially if the text contains commas, quotes, or line breaks.
              Example:
              text = \"[00.23s - 00.40s] This is a quote, with commas, 'apostrophes', and line breaks. Still safe."
        text_file: The path to the file where the quote will be saved, you have access to the variable, just write text_file=text_file.
    """
    with open(text_file, "a", encoding="utf-8") as f:
            f.write("===START_TEXT===\n")
            f.write(text.strip() + "\n")
            f.write("===END_TEXT===\n")
            print(f"text: {text}")




def video_creation_worker():
     while True:
          try:
             video_url,  final_start_time, final_end_time, subtitle_text = video_task_que.get()
          except queue.Empty:
                continue  
          try:
             log(f"\n\n\n\n\n\n\n\n\n\n[video_creation_worker] Current work being proccessed...[ video_url: {video_url}, start_time: {final_start_time}, end_time: {final_end_time}, text: {subtitle_text} to que]")
             log(f"[video_creation_worker] Processing video task: {video_url}, {final_start_time}-{final_end_time}")
             run_video_short_creation_thread(video_url, final_start_time, final_end_time, subtitle_text)
             log(f"Done Creating Video \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
          except Exception as e:
               log(f"[video_creation_worker] error during video_creation_worker: {str(e)}")
               raise ValueError(f"[video_creation_worker]Error during video creation")
          finally:
               print("HIT FINALLY BLOCK...")
             # video_task_que.task_done()





def wait_for_proccessed_video_complete(queue: Queue, check_interval=60):
     """Blocks until the queue is empty, checking every `check_interval` seconds."""
     log(f"\n\n\n\n\n\n[wait_for_proccessed_video_complete]")
     while not queue.empty():
          log(f"[wait_for_proccessed_video_complete]  waiting for video_task_que to be empty: items remaining: {queue.qsize()}")
          time.sleep(check_interval)
     log("[wait_for_proccessed_video_complete]✅ video_task_que is now empty.")









def verify_start_time_end_time_text(audio_path, Video_start_time, Video_end_time, text):
    """
    Verifies the correct absolute start/end time for the given text 
    by re-transcribing the clipped audio, matching the first and last word,
    and adjusting by the block's absolute start time.
    """

    from Extra_utils import SpeechToTextTool_verify
    import re
    import os
    import ffmpeg
    import tempfile

    log(f"[verify_start_time_end_time_text] Starting verification for text: '{text}'")
    log(f"[verify_start_time_end_time_text] Audio path: {audio_path}")
    log(f"[verify_start_time_end_time_text] Initial video start time: {Video_start_time}, end time: {Video_end_time}")

    tool = SpeechToTextTool_verify()
    tool.device = "cuda"
    tool.setup()
    log("[verify_start_time_end_time_text] SpeechToTextTool_verify initialized and set to CUDA")

    def clean_word(w):
        cleaned = re.sub(r"[^\w]", "", w.strip().lower())
        log(f"[clean_word] Original: '{w}' Cleaned: '{cleaned}'")
        return cleaned

    def remove_timestamps(text):

        cleaned_text = re.sub(r"\[\d+(\.\d+)?s\s*-\s*\d+(\.\d+)?s\]", "", text)
        log(f"[remove_timestamps] Tekst uten tidsstempler:\n{cleaned_text}")
        return cleaned_text

    def find_start_end_by_first_last_word(whisper_words, fasit_text, block_absolute_start):
        log(f"[find_start_end_by_first_last_word] Mottatt fasit_text: '{fasit_text}'")
        
      
        cleaned_text = remove_timestamps(fasit_text)
        fasit_words_cleaned = [clean_word(w) for w in cleaned_text.strip().split()]
        fasit_words = cleaned_text.strip().split()
        log(f"[find_start_end_by_first_last_word] Split fasit_words (uten tidsstempler): {fasit_words}")

        first_word = clean_word(fasit_words[0])
        last_word = clean_word(fasit_words[-1])
        log(f"[find_start_end_by_first_last_word] Første ord i fasit: '{first_word}', siste ord i fasit: '{last_word}'")

        relative_start = None
        relative_end = None

        for i, w in enumerate(whisper_words):
            w_word = clean_word(w['word'])
            log(f"[find_start_end_by_first_last_word] Sjekker whisper ord #{i}: '{w_word}', start: {w['start']}, end: {w['end']}")
            if relative_start is None and w_word == first_word:
                relative_start = float(w['start'])
                log(f"[find_start_end_by_first_last_word] Fant første ord '{first_word}' starttid: {relative_start}")
            if w_word == last_word:
                relative_end = float(w['end'])
                log(f"[find_start_end_by_first_last_word] Fant siste ord '{last_word}' sluttid: {relative_end}")

        if relative_start is None:
            relative_start = float(whisper_words[0]['start'])
            log(f"[find_start_end_by_first_last_word] Fant ikke første ord '{first_word}', bruker starttid for første whisper-ord: {relative_start}")
        if relative_end is None:
            relative_end = float(whisper_words[-1]['end'])
            log(f"[find_start_end_by_first_last_word] Fant ikke siste ord '{last_word}', bruker sluttid for siste whisper-ord: {relative_end}")

        absolute_start = block_absolute_start + relative_start
        absolute_end = block_absolute_start + relative_end

     
        subtitle_text = []
        fasit_idx = 0

        for w in whisper_words:
            if fasit_idx >= len(fasit_words_cleaned):
                break
            if clean_word(w['word']) == fasit_words_cleaned[fasit_idx]:
                subtitle_text.append(w)
                fasit_idx += 1

        log(f"[find_start_end_by_first_last_word] Beregner absolutt starttid: block_absolute_start ({block_absolute_start}) + relative_start ({relative_start}) = {absolute_start}")
        log(f"[find_start_end_by_first_last_word] Beregner absolutt sluttid: block_absolute_start ({block_absolute_start}) + relative_end ({relative_end}) = {absolute_end}")
        log(f"COMPLETE subtitle_text: {subtitle_text}")
        return absolute_start, absolute_end,subtitle_text


    def extract_audio_segment(audio_path, start_time, end_time):
        """
        Extracts audio segment from start_time to end_time.
        """
        duration = end_time - start_time
        log(f"[extract_audio_segment] Extracting audio from {start_time} to {end_time}, duration: {duration} seconds")
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        (
            ffmpeg
            .input(audio_path, ss=start_time, t=duration)
            .output(temp_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(quiet=True)
        )
        log(f"[extract_audio_segment] Audio segment saved to temporary file: {temp_path}")
        return temp_path

    temp_audio_path = extract_audio_segment(audio_path, Video_start_time, Video_end_time)

    try:
        whisper_results = tool.forward({"audio": temp_audio_path})
        log(f"[verify_start_time_end_time_text] Whisper transcription results: {whisper_results}")
        if not whisper_results:
            raise ValueError("Whisper result is empty or invalid")
    finally:
        os.remove(temp_audio_path)
        log(f"[verify_start_time_end_time_text] Temporary audio file deleted: {temp_audio_path}")

    whisper_words = whisper_results[0]['words']
    log(f"[verify_start_time_end_time_text] Extracted whisper words: {whisper_words}")

    final_start_time, final_end_time, subtitle_text = find_start_end_by_first_last_word(
        whisper_words, text, Video_start_time
    )


    log(f"[verify_start_time_end_time_text] Final absolute start time: {final_start_time}, end time: {final_end_time}")

    return final_start_time, final_end_time,subtitle_text






@tool
def create_motivationalshort(text: str) -> None:
        """
        Tool that creates a motivational shorts video.

        Args:
            text (str): The input text for the motivational short. It must include a timestamped quote in the format:
                ===START_TEXT===
                [start_time - end_time] actual text here.
                ===END_TEXT===
        """
        log(f"\n[CREATE_MOTIVATIONALSHORT]   text sent in: {text}")
        try:
            audio_path = get_current_audio_path()
            log(f"audio_path: {audio_path}")

            log(f"text before parse_multiline_block: {text}")
            start_time, end_time, new_text = parse_multiline_block(text)
            log(f"after [parse_multiline_block] --> start_time: {start_time}, end_time: {end_time}, new_text: {new_text}")
        except Exception as e:
             log(f" [verify_start_time_end_time_text]  Error during [parse_multiline_block]: {str(e)}")
                
        

        try:
            final_start_time,final_end_time, subtitle_text  = verify_start_time_end_time_text(audio_path=audio_path,text=new_text, Video_start_time=start_time,Video_end_time=end_time)
            log(f"[verify_start_time_end_time_text]final_start_time: {final_start_time}, final_end_time: {final_end_time}. final_text: {subtitle_text}")
        except Exception as e:
            log(f"[verify_start_time_end_time_text] error during verifying time and text: {str(e)}")


        if start_time is None or end_time is None:
             log("[verify_start_time_end_time_text] start_time is None or end_time is None")
             raise ValueError(f"start_time or end_time is None, start_time: {start_time}, end_time: {end_time}")
        

        log(f"[verify_start_time_end_time_text]  text: {new_text}, \n start_time: {start_time}, end_time: {end_time}")
        
       
        video_url = get_current_videourl()
        log(f"[verify_start_time_end_time_text] Starting video creation for {video_url} from {start_time}s to {end_time}s")

        try:
            print(f"[verify_start_time_end_time_text] Queued video task: url={video_url}, start={start_time}, end={end_time}, text='{text}'")
            video_task_que.put((video_url, final_start_time, final_end_time, subtitle_text))
            Delete_rejected_line(text)
            global count 
            count +=1
            
        except Exception as e:
             log(f"Error addng to queue: {str(e)}")


        

@tool
def Delete_rejected_line(text: str) -> None:
        """  Deletes a block from the text file that contains the given inner text,
             by removing the whole block: ===START_TEXT=== ... ===END_TEXT===
        Args:
            text: The line to delete (i.e., considered rejected/not valid) Format: 
              ===START_TEXT=== 
              [start_time - end_time] actual text here. 
              ===END_TEXT===      
        """
        log(f"\n[Delete_rejected_line] text into func: {text}")
        text_file = get_current_textfile()
        log(f"[Delete_rejected_line] path to textfile: {text_file}")

        with open(text_file, 'r', encoding="utf-8") as f:
                    content = f.readlines()

        block_text = text.strip() 

        match = re.search(r'===START_TEXT===\s*(.*?)\s===END_TEXT===',block_text, flags=re.DOTALL)
        if not match:
           log("[Delete_rejected_line] Could not extract block content — check input format!")
           return
        
        
        inner_text = re.escape(match.group(1).strip())


        pattern = rf'===START_TEXT===\s*.*?{inner_text}.*?\s===END_TEXT==='

        new_content, count = re.subn(pattern, '', content, flags=re.DOTALL)

        log(f"[Delete_rejected_line] Removed {count} block(s).\nNew content:\n{new_content.strip()}")

        
        with open(text_file, 'w', encoding="utf-8") as f:
            f.write(new_content.strip() + "\n")






def verify_saved_text_agent(agent_saving_path):
    set_current_textfile(agent_saving_path)
    print(f"agent_saving_path: {agent_saving_path}")
    global Global_model
    loaded_verify_saved_text_prompt = r'C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Prompt_templates\Verify_saved_quotes.yaml'
    with open(loaded_verify_saved_text_prompt, 'r', encoding='utf-8') as f:
            Prompt_template = yaml.safe_load(f)

    final_answer = FinalAnswerTool()
    create_motivational_short_agent = CodeAgent(
        model=Global_model,
        tools=[create_motivationalshort,Delete_rejected_line,final_answer],
        max_steps=1,
        prompt_templates=Prompt_template,
        stream_outputs=True
    )

    with open(agent_saving_path, "r", encoding="utf-8") as f:
             saved_quotes_text = f.read()

    task = f"""Analyze all the lines but do it line for line, step by step, 
    reject the lines that are not valid/suitable for a standalone motivational shorts video by using `Delete_rejected_line` tool and run  `create_motivationalshort` tool for each of those that are valid 
    now start step by step chain of thought reasoning over the lines:
    [{saved_quotes_text}] 
    """

    create_motivational_short_agent.run(task=task)
    del create_motivational_short_agent

def save_full_io_to_file(input_chunk: str, reasoning_steps: list[str], model_response: str, file_path: str) -> None:
    with open(file_path, "a", encoding="utf-8") as f:
        f.write("===INPUT CHUNK START===\n")
        f.write(input_chunk.strip() + "\n")
        f.write("===INPUT CHUNK END===\n\n")

        f.write("===REASONING STEPS===\n")
        for step in reasoning_steps:
            f.write(step + "\n")
        f.write("\n")

        f.write("===MODEL RESPONSE START===\n")
        f.write(model_response.strip() + "\n")
        f.write("===MODEL RESPONSE END===\n\n")
        f.write("------------------------------------------------------------------------\n\n\n")
















#Agent som analyserer tekst  fra transkript ved og lese (chunk for chunk) --->  (lagrer teksten basert på (task)) #eksempel her er motiverende/quote/inspirerende
def Transcript_Reasoning_AGENT(transcripts_path,agent_txt_saving_path):
    log(f"✅ Entered Transcript_Reasoning_AGENT() transcript_path: {transcripts_path}, agent_txt_saving_path: {agent_txt_saving_path}")
    ModelCountRun = 0
    global Global_model
    loaded_reasoning_agent_prompts = r'C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Prompt_templates\test.yaml'
    with open(loaded_reasoning_agent_prompts, 'r', encoding='utf-8') as f:
            Prompt_template = yaml.safe_load(f)

    final_answer = FinalAnswerTool()
    Reasoning_Text_Agent = CodeAgent(
        model=Global_model,
        tools=[SaveMotivationalText,final_answer],
        max_steps=1,
        verbosity_level=1,
        stream_outputs=True,
        prompt_templates=Prompt_template,
    )
    chunk_limiter = ChunkLimiterTool()

    log(f"transcript_path that is being proccessed inside func[Transcript_Reasoning_Agent]: {transcripts_path}")
    transcript_title = os.path.basename(transcripts_path)
    log(f"transcript title: {transcript_title}")
    log(f"\nProcessing new transcript: {transcripts_path}")
    with open(agent_txt_saving_path, "a", encoding="utf-8") as out:
        out.write(f"\n--- Transcript Title: {transcript_title} ---\n")
   
        
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
                print("Finished processing current transcript. Now exiting func [Transcript Reasoning Agent]")
                del Reasoning_Text_Agent
                break
        
        log(f"[The current ModelCountRun]: {ModelCountRun}")
        if  ModelCountRun >= 1:
                verify_saved_text_agent(agent_txt_saving_path)
                wait_for_proccessed_video_complete(video_task_que)
                ModelCountRun = 0

       
        task = f"""
                    You are an expert at identifying  powerful, share-worthy snippets from motivational podcast transcripts.
                    Your job is to:
                         1. Read the transcript chunk below and internally reason through its overall message.
                         2. Understand the connection between the lines. and what they are saying. This will happend internally and not in 'Thought: ' sequence.
                         Remember do not save any text that does not provide a complete thought. the goal is that this text will be used as a motivational shorts video. before you save the text you identified,  ask yourself if you were a listener, would you understand it.
                    2. Extract only those lines or passages that:
                        • Stand alone with full context (no missing setup).  
                        • Pack a punch of advice, insight, or inspiration.  
                        • Are memorable enough to anchor a motivational short video.
                        • Are complete thoughts or sentences, that if the text you decide to save were isolated from the rest would provide a complete thought and understanding for the listener.
                        • A complete thought is that the overall meaning of the setence/text does not miss any context like exsample of lacking context is that it starts with (and, but, etc).

                    Do NOT save generic fluff—the transcript as text in chunk is already motivational.
                    the text you choose too save needs to be complete and would result in a max  10-20 seconds motivational shorts video 
                    IF you no text is identified. nothing that could be a standalone moitvational short, only provide `final_answer` tool stating that, this will also successfully achieve the task.
                    Always confirm that the text saved also include the exact timestamps [start time - End time] text...

                    In the 'Thought: ' sequence. write a short summary describing the overall context of the chunk and then explain shortly what text you are looking for to save in order to fufill the task then procceed in 'Code: ' sequence  with the text to save after you internally have analyzed it all.
 

                    You must have analyzed, Focus on logical flow, completeness, and independence like a human reader on the text in chunk before any saving. and then save all identified text if any is present else provide only `final_answer`


                    Here is the chunk/text you will analyze:

                     [chunk start]\n
                      {chunk}  
                    \n[chunk end]  
                      """

        ModelCountRun += 1
        result = Reasoning_Text_Agent.run(
                task=task,
                additional_args={"text_file": agent_txt_saving_path}
            )
        print(f"[Path to where the [1. reasoning agent ] saves the motivational quotes  ]: {agent_txt_saving_path}")
        print(f"Agent response: {result}\n")
        save_full_io_to_file(
            input_chunk=chunk,
            reasoning_steps=reasoning_log,
            model_response=result,
            file_path=r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\debug_performance\AgentRun_Data.txt"
        )

        chunk_limiter.called = False 











log_file_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\debug_performance\transcription_log.txt"

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    thread_name = threading.current_thread().name
    log_message = f"[{timestamp}][{thread_name}] {msg}"
    print(log_message)
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")


video_creation_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\debug_performance\Video_creationLog_path.txt"
def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    thread_name = threading.current_thread().name
    log_message = f"[{timestamp}][{thread_name}] {msg}"
    print(log_message)
    with open(video_creation_path, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")

def transcribe_single_video(video_path, device):
    log(f"Starting transcription for {video_path} on device {device}")

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
        transcript_queue.put((video_path, txt_output_path,agent_text_saving_path))
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
        set_current_audio_path(audio_path)
        log(f"Extracted audio → {audio_path}")
    except subprocess.CalledProcessError:
        log(f"❌ Audio extraction failed for {video_path}")
        return

    try:
        start_time = time.time()  
        if device == "cuda":
                got_gpu = gpu_lock.acquire(blocking=False)
                if not got_gpu:
                    log(f"GPU busy, falling back to CPU for {video_path}")
                    device = "cpu"
                else:
                    log(f"Acquired GPU lock for {video_path}")
        else:
             got_gpu = False  

   
       
        tool = SpeechToTextTool() if device == "cuda" else SpeechToTextToolCPU()
        tool.device = device
        tool.setup()

        try:
             print(f"starting TRANSCRIPTION")
             result_txt_path = tool.forward({"audio": audio_path, "text_path": txt_output_path, "video_path": video_path})
        except Exception as e:
             print(f"error during transcribing: {str(e)}")
        elapsed_time = time.time() - start_time 
        log(f"⏱️ Transcription took {elapsed_time:.2f} seconds for {video_path} on device {device}")

        if result_txt_path != txt_output_path:
            os.rename(result_txt_path, txt_output_path)
        log(f"🔊 Transcription saved → {txt_output_path}")

        transcript_queue.put((video_path, txt_output_path, agent_text_saving_path))
        gpu_lock.release()
        del tool     
        log(f"Released GPU lock after transcribing {video_path}\n")
        log(f"added video_path: {video_path}, transcript: {txt_output_path}, agent_saving_path: {agent_text_saving_path} to queue  for GPU processing")

    except Exception as e:
        log(f"Transcription failed for {audio_path}: {e}")

def gpu_worker():
    log("GPU worker started")
    torch.backends.cudnn.benchmark = True

 
    global Global_model
    Global_model = TransformersModel(
            model_id = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Ministral-8B-Instruct-2410",
            load_in_4bit=True,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
            max_new_tokens=5000,
        )
    log(f"Loaded Global_model on device")
    itemcount = 0

    while True:
        item = transcript_queue.get()

        log(f"item: {item}")
        if item is None and itemcount <= 0:
            log("Shutdown signal received, exiting GPU worker")
            transcript_queue.task_done()
            break

        itemcount += 1

        video_path_url, transcript_text_path,agent_txt_saving_path = item
        set_current_videourl(video_path_url)
        log(f"Processing {video_path_url} & {transcript_text_path} in GPU worker, agent is using txt path to save: {agent_txt_saving_path}")

        log(f"GPU lock acquired by gpu_worker for {video_path_url}")

        try:
            Transcript_Reasoning_AGENT(transcript_text_path, agent_txt_saving_path)
            log(f"Transcript_Reasoning_AGENT has exited...")
        finally:
            log(f"GPU lock released by gpu_worker for {video_path_url}")
            transcript_queue.task_done()




if __name__ == "__main__":
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    worker_thread = threading.Thread(target=video_creation_worker)
    worker_thread.start()
    set_current_audio_path(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\work_queue_folder\The Formula for Thriving in a Changing World (Success, Intuition, & Fulfillment) ｜ Vishen Lahkiani\The Formula for Thriving in a Changing World (Success, Intuition, & Fulfillment) ｜ Vishen Lahkiani.wav")
    set_current_videourl(r"c:\Users\didri\Documents\The Formula for Thriving in a Changing World (Success, Intuition, & Fulfillment) ｜ Vishen Lahkiani.mp4")
    create_motivationalshort(text="[4732.11s - 4733.69s] only [4733.69s - 4734.31s] our health[4734.31s - 4735.49s] but so many")

    # video_paths = [
    #     r"c:\Users\didri\Documents\The Formula for Thriving in a Changing World (Success, Intuition, & Fulfillment) ｜ Vishen Lahkiani.mp4",
    # ]
    # log(f"Video_paths: {len(video_paths)}")

    # devices = ["cuda", "cpu"] 
    # video_device_pairs = [(video_paths[i], devices[i % len(devices)]) for i in range(len(video_paths))]

    # max_threads = 1
    # start_time = time.time()
    # with ThreadPoolExecutor(max_workers=max_threads) as executor:
    #     futures = []
    #     start_time = time.time()
    #     for video_path, device in video_device_pairs:
    #         futures.append(executor.submit(transcribe_single_video, video_path, device))

    #     for future in futures:
    #         future.result() 
    #         end_time = time.time()
    #     total_time = end_time - start_time
    #     log(f"start_time of threadpool: {start_time} & endtime of threadpool = {end_time}, total: {total_time}")


    # gpu_thread = threading.Thread(target=gpu_worker, name="GPU-Worker")

    # gpu_thread.start()


    # transcript_queue.join() 
    # gpu_thread.join()
    # worker_thread.join()


