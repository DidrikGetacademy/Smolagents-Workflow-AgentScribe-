import sys
import os
REPO_ROOT = os.path.abspath('.')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import json
import datetime
GPEN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'GPEN'))
if GPEN_PATH not in sys.path:
    sys.path.insert(0, GPEN_PATH)
import GPEN.__init_paths
from GPEN.face_enhancement import FaceEnhancement
from utility.Custom_Agent_Tools import SpeechToTextToolCUDA, SpeechToText_short_creation_thread,Background_Audio_Decision_Model
import gc
from utility.log import log
import torchvision.transforms.functional as F
sys.modules['torchvision.transforms.functional_tensor'] = F
import torch.nn.functional as F
import subprocess
from moviepy.audio.fx import MultiplyVolume
from moviepy import VideoFileClip, ImageSequenceClip, TextClip, CompositeVideoClip,vfx,AudioFileClip,afx,CompositeAudioClip,afx
import cv2
import ffmpeg
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
from blender.blender import enhance_frames_bpy
from pydub import AudioSegment
from proglog import ProgressBarLogger
import utility.Global_state as Global_state
from queue import Queue
from utility.create_montage_short import _create_montage_short_func, compose_montage_clips
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







#-----------------------------------#
# Shorts Video Creation/Functions
#-----------------------------------#
def change_brightness(frame, amount=0.2):
    """
    Darken or brighten the frame.
    amount > 0 : brighten
    amount < 0 : darken
    Optimized to avoid float32 conversion
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    # Work with V channel as int16 to avoid overflow, then clip
    v_channel = hsv[..., 2].astype(np.int16)
    v_channel = np.clip(v_channel * (1.0 + amount), 0, 255).astype(np.uint8)
    hsv[..., 2] = v_channel
    changed_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return changed_frame


def change_saturation(frame, mode="Increase", amount=0.2):
    """
    Change saturation without float32 conversion to save memory.
    Uses int16 temporarily to avoid overflow.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    if mode == "Grayscale":
        hsv[..., 1] = 0  # Set saturation to 0 for completely gray frames
    else:
        # Work with S channel as int16 to avoid overflow during multiplication
        s_channel = hsv[..., 1].astype(np.int16)
        if mode == "Increase":
            s_channel = np.clip(s_channel * (1.0 + amount), 0, 255).astype(np.uint8)
        elif mode == "Decrease":
            s_channel = np.clip(s_channel * (1.0 - amount), 0, 255).astype(np.uint8)
        hsv[..., 1] = s_channel

    # Convert back to RGB
    changed_frames = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return changed_frames




def _resize_smart(img, target_w: int, target_h: int):
    """Resize with better interpolation:
    - INTER_AREA when downscaling
    - INTER_CUBIC when upscaling
    """
    h, w = img.shape[:2]
    if target_w < w or target_h < h:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC
    return cv2.resize(img, (target_w, target_h), interpolation=interp)


def mix_audio(original_audio, background_music_path, bg_music_volume=0.25):
        bg_music = AudioFileClip(background_music_path)
        if bg_music.duration < original_audio.duration:
            bg_music = bg_music.with_effects([afx.AudioLoop(duration=original_audio.duration)])
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
        input_name = session.get_inputs()[0].name
        print("ONNX input shape:", session.get_inputs()[0].shape)
        progress_bar = tqdm(total=len(frames), desc="[detect_and_crop_frames_batch]Processing frames", unit="frame", dynamic_ncols=True)
        try:
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i+batch_size]
                original_count = len(batch)


                if len(batch) < batch_size:
                    pad_count = batch_size - len(batch)
                    batch += [np.zeros_like(batch[0])] * pad_count


                processed_batch = []
                for frame in batch:

                    img = cv2.resize(frame, (928, 928), interpolation=cv2.INTER_AREA)
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
                        cropped_frame = _resize_smart(cropped_frame, TARGET_W, TARGET_H)
                    cropped_frames.append(cropped_frame)
        finally:
            progress_bar.close()
            del  predictions, detections, frame, det, h, w, areas, max_idx,session
            if batch is not None:
                del batch
            session = None
            clean_get_gpu_memory(threshold=0.1)
        return cropped_frames



#------------------------------------------------------------------------------------------------------------------------#
# create_short_video --> Function takes (video, start time/end time for video, video name, subtitles for video) as input
#------------------------------------------------------------------------------------------------------------------------#
def create_short_video(video_path, audio_path, start_time, end_time, video_name, subtitle_text,Video_output_path=None):
    probe = ffmpeg.probe(video_path)
    log(probe)
    format_info = probe.get('format', {})
    video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
    bitrate = int(format_info.get('bit_rate', 0))
    video_codec = video_streams[0]['codec_name'] if video_streams else None
    audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']
    audio_codec = audio_streams[0]['codec_name'] if audio_streams else None

    def group_subtitle_words_in_triplets(subtitle_words):
         chunks = []
         if not subtitle_words:
             return chunks

         offset = float(subtitle_words[0]['start'])
         segments = []
         current_segment = [subtitle_words[0]]
         PAUSE_THRESHOLD = 0.25

         for i in range(1, len(subtitle_words)):
              prev_word = subtitle_words[i-1]
              curr_word = subtitle_words[i]
              gap = float(curr_word['start']) - float(prev_word['end'])

              if gap > PAUSE_THRESHOLD:
                  current_segment[-1]['end'] += 0.2
                  segments.append(current_segment)
                  current_segment = []

              current_segment.append(curr_word)

         if current_segment:
              segments.append(current_segment)

         MAX_WORDS_PER_CHUNK = 5

         for segment in segments:
              for i in range(0, len(segment), MAX_WORDS_PER_CHUNK):
                   chunk_words = segment[i : i + MAX_WORDS_PER_CHUNK]

                   text_chunk = ''.join([w['word'].strip() + ' ' for w in chunk_words]).strip().upper()
                   start = float(chunk_words[0]['start']) - offset
                   end = float(chunk_words[-1]['end']) - offset
                   duration = max(0.0, end - start)
                   start = max(0.0, start)

                   chunks.append({'text': text_chunk, 'start': start, 'end': end, 'duration': duration})

         return chunks

    try:
       triplets = group_subtitle_words_in_triplets(subtitle_text)
    except Exception as e:
         log(f"[group_subtitle_words_in_triplets] Error during grouping of subtitles in triplets. {str(e)} ")


    def create_subtitles_from_triplets(triplets):
        text_clips = []

        for i, c in enumerate(triplets):

            _text = c['text']
            _text = _text.upper()
            log(f"text: {_text}")
            txt_clip = TextClip(
                text=_text,
                font=r"C:\WINDOWS\FONTS\COPPERPLATECC-BOLD.TTF",
                font_size=35,
                margin=(10, 10),
                text_align="center",
                vertical_align="center",
                horizontal_align="center",
                color='white',
                stroke_color="black",
                stroke_width=2,
                size=(1080, 300),
                method="label",
                duration=c['duration']
            ).with_start(c['start']).with_position(('center', 0.44), relative=True)


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
    cropped_frames = detect_and_crop_frames_batch(frames=frames,batch_size=8)
    clean_get_gpu_memory(threshold=0.1)
    del frames







# --------------------------------------#
#  Blender version 5.0.0 detail enahcement
# ----------------------------------------#
    # log(f"blender frames proccessing now.")
    # gc.collect()
    # gc.collect()
    # blender_frames = enhance_frames_bpy(cropped_frames, batch_size=10,use_skin_mask=True)
    # del cropped_frames
    # clean_get_gpu_memory(threshold=0.1)






# ----------------------#
#   FACEENCHANCEMENT
# ---------------------#
    class Face_enchance_Args:
            model = 'GPEN-BFR-2048'
            task = 'FaceEnhancement'
            key = None
            in_size = 2048
            out_size = 0
            channel_multiplier = 2
            narrow = 1
            alpha = 0.5
            use_sr = True
            use_cuda = True
            save_face = False
            aligned = True
            sr_model = 'realesrnet'
            sr_scale = 2
            tile_size = 0
            ext = '.png'
    face_args = Face_enchance_Args()
    Skin_texture_enchancement = FaceEnhancement(
        face_args,
        in_size=face_args.in_size,
        model=face_args.model,
        use_sr=face_args.use_sr,
        device='cuda'
    )
    FaceEnhancement_frames = []
    try:
         for f_frames in tqdm(cropped_frames, desc="[FaceEnhancement]  proccessing frames", unit="frame"):
              frame_height, frame_width = f_frames.shape[:2]
              bgr_frame = cv2.cvtColor(f_frames, cv2.COLOR_RGB2BGR)
              enchanced_frame_bgr, _, _ = Skin_texture_enchancement.process(bgr_frame)
              RGB_face_enchanced_frame = cv2.cvtColor(enchanced_frame_bgr, cv2.COLOR_BGR2RGB)
              FaceEnhancement_frames.append(RGB_face_enchanced_frame)
              del enchanced_frame_bgr,RGB_face_enchanced_frame
              gc.collect()
         del  Skin_texture_enchancement,bgr_frame,#blender_frames
         clean_get_gpu_memory(threshold=0.1)
    except Exception as e:
            log(f"[FaceEnhancement] Error: {str(e)}")









#-------------------------------#
# Creating videoclip from frames
#-------------------------------#
    try:
       processed_clip = ImageSequenceClip(FaceEnhancement_frames, fps=clip.fps).with_duration(clip.duration)
       clean_get_gpu_memory(threshold=0.1)

       gc.collect()
    except Exception as e:
         log(f"[processed_clip] ERROR: {str(e)}")



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


        if Latest_Yt_channel == "MA_Youtube":
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
        elif Latest_Yt_channel == "MR_Youtube":
                YT_channel = "MA_Youtube"
                overlay_clip = VideoFileClip(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\alpha_mask\YT_logo\LOGO_MA.mp4",has_mask=True)
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
        log(f"Error:  {str(e)}")


    try:
        subtitle_clips = create_subtitles_from_triplets(triplets)
    except Exception as e:
            log(f"error during [create_subtitles_from_triplets]: {str(e)}")



#-----------------------------------------------#
# Adds Subtitles + Logo to the video
#-----------------------------------------------#
    final_clip = CompositeVideoClip(
                [processed_clip.with_position('center')]  + subtitle_clips +  [logo_with_mask.with_position('center',0.50)] ,
                size=processed_clip.size
                )

    del logo_with_mask, subtitle_clips, processed_clip
    clean_get_gpu_memory(threshold=0.1)


    fade = CrossFadeIn(1.5)
    final_clip = fade.apply(final_clip)

# -------------------------------#
# Adds Background Music to video
# -------------------------------#
    log("choosing Background Audio\n")
    try:
        already_uploaded_videos = f"Video_clips/Youtube_Upload_folder/{YT_channel}/already_uploaded.txt"
        log(f"already_uploaded_videos: {already_uploaded_videos}")
        result = Background_Audio_Decision_Model(audio_file=audio_path,video_path=video_path,already_uploaded_videos=already_uploaded_videos,start_time=start_time,end_time=end_time)
        if os.path.exists(audio_path):
            os.remove(audio_path)
            log(f"Removed temporary audio file: {audio_path}")

        background_audio = result.get("path", "")
        song_name = result.get("song_name", "")
        Background_Audio_Reason = result.get("reason", "")
        log(f"Background Audio : {background_audio}, song_name: {song_name}, Background_Audio_Reason: {Background_Audio_Reason}")
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



    clean_get_gpu_memory(threshold=0.1)





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
        "-vf", "minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd_threshold=50",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        "-movflags", "+faststart",
    ],
    audio_bitrate="384k",
    remove_temp=True
        )


    gc.collect()
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
        clean_get_gpu_memory(threshold=0.1)





#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# A Agent creates optimized (Title, Description, Hashtags, Tags, category, publishAt) after analyzing similar trending videos related to input video &  Uploads the video to Youtube
# - Reloads the Global Model
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    try:
        global Global_model
        if 'Global_model' not in globals():
                Global_model = None
        from utility.upload_Socialmedia import upload_video
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
                Global_model = Reload_and_change_model(model_name="gpt-5-minimal", message="Reloading model to -> gpt-5-minimal before running [upload_video]")
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
            adjusted_start = max(0, video_start_time - padding_offset)
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
                    tool = SpeechToText_short_creation_thread()

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
               _create_montage_short_func(
                   video_path=video_path,
                   start_time=new_video_start_time,
                   end_time=new_video_end_time,
                   subtitle_text=crafted_Subtitle_text,
                   video_name=os.path.splitext(part_basename)[0],
                   Video_output_path=part_output_path,
                   YT_channel=YT_channel,
                   order=order,
                   middle_order=middle_order

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
                       output_path = compose_montage_clips(ordered_paths, final_output_path,YT_channel,_truncate_audio,video_path)
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


    parent_folder = os.path.join(script_dir, "work_queue_folder/in_progress")
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

















if __name__ == "__main__":
    video_paths = [
    r"c:\Users\didri\Documents\Hacking Your Psychology to Do Hard Things Consistently - Dr Mike Israetel.mp4",
    r"c:\Users\didri\Documents\Give Me 30 Minutes and Finally STOP Feeling Behind in Life.mp4",
    r"c:\Users\didri\Documents\Give Me 30 Minutes and I'll Make You Confident & Remove ALL Your Self Doubt! with Jay Shetty.mp4",
    r"c:\Users\didri\Documents\8 Things To Tell Yourself Every Morning.mp4",
    r"C:\Users\didri\Documents\Confidenco.mp4",
    r"c:\Users\didri\Documents\6 LIFE-CHANGING lessons I've learned from EXTRAORDINARY Guests I Wish I knew Sooner….mp4",
    r"c:\Users\didri\Documents\8 Things To Tell Yourself Every Morning.mp4",
    r"c:\Users\didri\Documents\A Toolkit for Confidence： How to Build UNSHAKABLE Self Confidence ｜ The Mel Robbins Podcast.mp4",
    r"c:\Users\didri\Documents\Change Your Brain： Neuroscientist Dr. Andrew Huberman ｜ Rich Roll Podcast.mp4",
    r"c:\Users\didri\Documents\Daily Habits for Increasing Grit & Resilience ｜ Michael Easter & Dr. Andrew Huberman.mp4",
    r"c:\Users\didri\Documents\Focus on Yourself, Not Others ｜ Jim Rohn Mindset.mp4",
    r"c:\Users\didri\Documents\Give Me 27 Minutes and I’ll End Your Perfectionism for Good (FINALLY Get Unstuck!).mp4",
    r"c:\Users\didri\Documents\How to Defeat Your Stress, Anxiety & Inaction - Mel Robbins.mp4",
    r"c:\Users\didri\Documents\If you’re ambitious but lazy, please watch this….mp4",
    r"c:\Users\didri\Documents\Jordan Peterson： STOP LYING TO YOURSELF! How To Turn Your Life Around In 2024!.mp4",
    r"c:\Users\didri\Documents\The Let Them Theory： How to Take Back Your Peace and Power.mp4",
    ]
    clean_get_gpu_memory(threshold=0.1)
    Clean_log_onRun()


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

