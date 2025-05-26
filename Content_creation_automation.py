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
import cv2
import ffmpeg
import time
from typing import List
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import argparse
from concurrent.futures import ThreadPoolExecutor
from SwinIR.models.network_swinir import SwinIR as net
from tqdm import tqdm


_current_video_url: str = None
def set_current_videourl(url: str):
    global _current_video_url
    _current_video_url = url


def get_current_videourl() -> str:
    global _current_video_url
    return _current_video_url


Chunk_saving_text_file = r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Logging_and_filepaths\saved_transcript_storage.txt"
Final_saving_text_file=r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Logging_and_filepaths\final_saving_motivational.txt"
#model_path_SwinIR_color_denoise15_pth = r"c:\Users\didri\Desktop\LLM-models\Video-upscale-models\SwinIR-M_noise15.pth"
model_path_SwinIR_color_denoise15_onnx = r"c:\Users\didri\Desktop\LLM-models\Video-upscale-models\SwinIR_color_denoise15.onnx"
#model_path_Swin_BSRGAN_X4_pth = r"c:\Users\didri\Desktop\LLM-models\Video-upscale-models\BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
model_path_Swin_BSRGAN_X4_onnx = r"c:\Users\didri\Desktop\LLM-models\Video-upscale-models\BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.onnx"

def parse_multiline_block(block_text):
    lines = [line.strip() for line in block_text.strip().splitlines() if line.strip()]
    line = [line for line in lines if line.startswith('[')]
    print(f"[parse_multiline_block] lines={lines}")
    print(f"[parse_multiline_block] line=[{line}]")
    if not lines:
        return None, None, 
    
    start_time, _ = parse_timestamp_line(line[0])
    print(f"start_time: {start_time}")

    _, end_time = parse_timestamp_line(line[-1])

    print(f"end_time: {end_time}")
        
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
        text: The quote or message to save, include all timestamp if [start - end]. To avoid syntax errors, wrap the string in triple quotes 
              when calling this function, especially if the text contains commas, quotes, or line breaks.
              Example:
              text = "This is a quote, advice to be saved.
        text_file: The path to the file where the quote will be saved.
    """
    with open(text_file, "a", encoding="utf-8") as f:
        f.write("New text saved:" + text.strip() +"\n\n")
        print(f"text: {text}")
        start_time, end_time = parse_multiline_block(text)
        print(f"[start_time: {start_time}, end_time: {end_time}]   FROM : SaveMotivationalQuote_CreateShort")
    
    if start_time is None or end_time is None:
        raise ValueError(f"Start_time or end_time is None, start_time: {start_time}, end_time: {end_time}")
  

    video_url = get_current_videourl()
    print("[VIDEO URL IN: SaveMotivationalQuote_CreateShort]", video_url)
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
              text = \"This is a quote, with commas, 'apostrophes', and line breaks. Still safe."
        text_file: The path to the file where the quote will be saved.
    """
    with open(text_file, "a", encoding="utf-8") as f:
            f.write("===START_QUOTE===\n")
            f.write(text.strip() + "\n")
            f.write("===END_QUOTE===\n\n")






class realesgran:
    def __init__(self, frames, device):
        self.model_path= r"c:\Users\didri\Desktop\LLM-models\Video-upscale-models\RealESRGAN_x2plus.pth"
        self.device = device
        self.frame_to_upscale = frames
        self.model =  RRDBNet(
            num_in_ch=3, 
            num_out_ch=3, 
            num_feat=64, 
            num_block=23, 
            num_grow_ch=32, 
            scale=2
        )
        self.real_esrgan = None
        

    def load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'params_ema' in checkpoint:
            self.model.load_state_dict(checkpoint['params_ema'], strict=True)
        else:
            self.model.load_state_dict(checkpoint['params'], strict=True)
        self.model.to(self.device)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        real_esrgan = RealESRGANer(
            scale=2,                 
            model_path=self.model_path,
            model=self.model,
            tile=0,                    
            tile_pad=10,
            pre_pad=0,
            half=True,              
            device=self.device
        )
      

      
    def upscale_frames(self):
        if self.real_esrgan is None:
            self.load_model()
        upscaled = []
        count = 0
        for frame in self.frame_to_upscale:
            count += 1
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output, _ = self.realreal_esrgan.enhance(img_rgb, outscale=2)
            out_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            print(f"appending upscaled frame: {out_bgr.shape} [COUNT]: {count}/{len(self.frames_to_upscale)}")
            upscaled.append(out_bgr)

        return upscaled


import os
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm
import onnxruntime as ort

class swinir_processor:
    def __init__(self, processed_frames, model_name):
        try:
            self.model_path = model_path_Swin_BSRGAN_X4_onnx if "x4_GAN" in model_name else model_path_SwinIR_color_denoise15_onnx
            self.model_name = model_name
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = None
            self.session = None
            self.use_onnx = self.model_path.endswith(".onnx")
            self.window_size = 8 if "noise15" in model_name else 7
            self.scale = 4 if "x4_GAN" in model_name else 1
            self.processed_frames = processed_frames
            self.args = argparse.Namespace(
                tile=256,
                tile_overlap=32,
                scale=self.scale
            )

        except Exception as e:
            print(f"Error in __init__: {str(e)}")
            raise

    def test(self, img_lq, model, args, window_size):
        try:
            def infer(x):
                if self.use_onnx:
                    ort_inputs = {self.session.get_inputs()[0].name: x.cpu().numpy()}
                    ort_outs = self.session.run(None, ort_inputs)
                    out_tensor = torch.from_numpy(ort_outs[0]).to(x.device)
                    return out_tensor
                else:
                    return model(x)

            if args.tile is None:
                output = infer(img_lq)
            else:
                b, c, h, w = img_lq.size()
                tile = min(args.tile, h, w)
                assert tile % window_size == 0, "tile size should be a multiple of window_size"
                tile_overlap = args.tile_overlap
                sf = args.scale

                stride = tile - tile_overlap
                h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
                w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
                E = torch.zeros(b, c, h * sf, w * sf).type_as(img_lq)
                W = torch.zeros_like(E)

                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        in_patch = img_lq[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                        out_patch = infer(in_patch)
                        out_patch_mask = torch.ones_like(out_patch)

                        E[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch)
                        W[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch_mask)
                output = E.div_(W)
            return output
        except Exception as e:
            print(f"Error in test(): {str(e)}")
            raise

    def return_model(self):
        try:
            if self.use_onnx:
                providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
                sess_options = ort.SessionOptions()
                sess_options.log_severity_level = 0  
                self.session = ort.InferenceSession(self.model_path, sess_options, providers=providers)

                print("[INFO] Loaded ONNX model.")
                return None
            else:
                if "x4_GAN" in self.model_path:
                    print("swinir model being used is [x4_GAN]")
                    model = net(upscale=self.scale, in_chans=3, img_size=64, window_size=8,
                                img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                                num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                                mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
                    param_key_g = 'params_ema'
                elif "M_noise15" in self.model_path:
                    print("swinir model being used is [M_noise15]")
                    model = net(upscale=1, in_chans=3, img_size=128, window_size=8,
                                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                                num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
                                upsampler='', resi_connection='1conv')
                    param_key_g = 'params'
                else:
                    raise ValueError(f"Unknown model path or model type: {self.model_path}")

                pretrained_model = torch.load(self.model_path)
                model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
                return model
        except Exception as e:
            print(f"Error in return_model(): {str(e)}")
            raise

    def run_inference(self):
        try:
            print(f"self.device: --> {self.device}")
            self.model = self.return_model()
            if self.model:
                self.model.eval()
                self.model = self.model.to(self.device)
                print(f"[INFO] Model loaded to device: {self.device}")
                print(f"[INFO] Model dtype: {next(self.model.parameters()).dtype}")

        except Exception as e:
            print(f"Error loading model in run_inference(): {str(e)}")
            return ValueError("error during model initialization")

        upscaled_frames = []
        total_frames = len(self.processed_frames)
        print(f"Starting inference on {total_frames} frames...")
        for i, frame in enumerate(tqdm(self.processed_frames, desc="processing frames", unit="frame")):
            try:
                upscaled_frame = self.process_frame(frame)
                upscaled_frames.append(upscaled_frame)
            except Exception as e:
                print(f"Error processing frame {i} in run_inference(): {str(e)}")
                raise
        print("Inference completed successfully.")
        
        downscaled_frames = downscale_to_size(upscaled_frames)
        
        sharpened_frames = sharpen_frame_naturally(downscaled_frames)
    
        return sharpened_frames

    def process_frame(self, frame):
        try:
            img_lq = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_lq = img_lq.astype(np.float32) / 255.0
            img_lq = np.transpose(img_lq, (2, 0, 1))
            img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)

            _, _, h_old, w_old = img_lq.shape
            h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
            w_pad = (w_old // self.window_size + 1) * self.window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]

            with torch.no_grad():
                output = self.test(img_lq, self.model, self.args, self.window_size)
                output = output[..., :h_old * self.scale, :w_old * self.scale]

            output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output, (1, 2, 0))
            output = output[..., [2, 1, 0]]
            return (output * 255).astype(np.uint8)
        except Exception as e:
            print(f"Error in process_frame(): {str(e)}")
            raise


    

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




def create_short_video(video_path, start_time, end_time, video_name, subtitle_text):
    probe = ffmpeg.probe(video_path)
    print(probe)
    format_info = probe.get('format', {})
    bitrate = int(format_info.get('bit_rate', 0))
    video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
    video_codec = video_streams[0]['codec_name'] if video_streams else None
    audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']
    audio_codec = audio_streams[0]['codec_name'] if audio_streams else None

    

    subtitles = subtitle_text
    print("subtitles: ",subtitles)
    torch.cuda.set_device(0)
    Face_Detection_Yolov8x = r"c:\Users\didri\Desktop\LLM-models\Face-Detection-Models\yolov8x-face-lindevs.pt"
    print("Before loading YOLO model")
    try:
      model = YOLO(Face_Detection_Yolov8x) 
    except Exception as e:
        print(f"error loading model: {str(e)}")
    
    try:
        full_video = VideoFileClip(video_path)
        clip = full_video.subclipped(start_time, end_time)
        print(f"clip duration: {clip.duration}, clip fps: {clip.fps}, clip width: {clip.w}, clip height: {clip.h}, start_time: {start_time}, end_time: {end_time}, video_path: {video_path}")
    except Exception as e:
        print(f"full_video and clip  _video functions failed: {str(e)}")

    def split_subtitles_into_chunks(text, max_words=3):
        words = text.split()
        return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    
    def create_subtitles(txt,duration,clip_relative_start):
        chunks = split_subtitles_into_chunks(txt)
        chunk_duration = duration / len(chunks)
        text_clips = []
        for i, chunk in  enumerate(tqdm(chunks, desc="Processing chunk", unit="chunk")):
            start = clip_relative_start + i * chunk_duration

            print(duration)
            txt_clip = TextClip(
                text=chunk,
                font=r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Logging_and_filepaths\Video_clips\Cardo-Bold.ttf", 
                font_size=80,
                margin=(10, 10), 
                text_align="center" ,
                vertical_align="center",
                horizontal_align="center",
                color='white',
                stroke_color="black",
                stroke_width=2,
                size=(1000, None),
                method="label",
                duration=chunk_duration
            ).with_position(('center', 0.50), relative=True
            ).with_start(start)
            text_clips.append(txt_clip)
            print(f"appending: {txt_clip}")
            
        return text_clips
    
    
    def detect_and_crop_frames_batch(frames,batch_size=8):
        TARGET_W, TARGET_H = 1080, 1920
        alpha = 0.1
        prev_cx, prev_cy = None, None
        cropped_frames = []

        for i in range (0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            batch_imgs = [np.ascontiguousarray(f) for f in batch]

            results_batch = model(batch_imgs, imgsz=928)
            for frame, results in zip(batch, results_batch):
                face_boxes = [ box.xyxy.cpu().numpy().astype(int)[0] for box in results.boxes]
                h, w, _ = frame.shape
                aspect_ratio = TARGET_W / TARGET_H  
                if w / h > aspect_ratio:
                
                    crop_h = h
                    crop_w = int(h * aspect_ratio)
                else:
            
                    crop_w = w
                    crop_h = int(w / aspect_ratio)


                face_box = None
                if face_boxes:
            
                    face_areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in face_boxes]
                    max_face_idx = np.argmax(face_areas)
                    face_box = face_boxes[max_face_idx]

                if face_box is not None:
                    fx1, fy1, fx2, fy2 = face_box
                    cx, cy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
                    print(f"Detected face at ({cx}, {cy}), cropping around that.")

                else:
                    cx, cy = w // 2, h // 2  
                    print(f" no face found defaulting to center")

            

                if prev_cx is None or prev_cy is None:
                    sx, sy = cx, cy
                else:
                    sx = int(alpha * cx + (1 - alpha) * prev_cx)
                    sy = int(alpha * cy + (1 - alpha) * prev_cy)

                prev_cx, prev_cy = sx, sy

                x0 = max(0, min(cx - crop_w // 2, w - crop_w))
                y0 = max(0, min(cy - crop_h // 2, h - crop_h))

                cropped_frame = frame[y0:y0+crop_h, x0:x0+crop_w]

        
                if cropped_frame.shape[0] != TARGET_H or cropped_frame.shape[1] != TARGET_W:
                    cropped_frame = cv2.resize(cropped_frame, (TARGET_W, TARGET_H))

                cropped_frames.append(cropped_frame)
                print(f"appending {len(cropped_frames)} frames to list")

        return cropped_frames

    try:
        print("Starting to extract frames...")
        frames = []
        for frame in clip.iter_frames():
            frames.append(frame)
        print(f"Extracted {len(frames)} frames.")
        processed_frames = detect_and_crop_frames_batch(frames, batch_size=8)
        print("Cleared model from gpu")
        print(f"Processed frames = {len(processed_frames)}VIDEO PATH: {video_path}, START TIME: {start_time}, END TIME: {end_time}, VIDEONAME: {video_name}")
    except Exception as e:  
            print(f"Error during frame extraction or processing: {str(e)}")
            return 
    




    del model 
    try:
        swinir_denoise = swinir_processor(processed_frames, model_name="SwinIR-M_noise15")
        print("swinir_processor for denoising instantiated successfully")
        denoised_frames = swinir_denoise.run_inference()
        print("frames for denoising finished successfully")
        del swinir_denoise
    except Exception as e:
          print(f"error during swinir inference with model denoise15: {str(e)}")

    try:
        swinir_upscaler = swinir_processor(denoised_frames, model_name="x4_GAN")
        print("swinir_processor for upscaling instantiated successfully")
        upscaled_frames = swinir_upscaler.run_inference()
        print("frames for upscaling finished successfully")
        del swinir_upscaler
    except Exception as e:
        print(f"error during swinir inference with model X4: {str(e)}")

    
    processed_clip = ImageSequenceClip(upscaled_frames, fps=clip.fps).with_duration(clip.duration)
    
    subtitle_clips = []
    for text, start, end in subtitles:
        clip_relative_start = start - start_time
        clip_relative_end = end - start_time
        duration = clip_relative_end - clip_relative_start
        print(f"subtitle_clips: {subtitle_clips}")
        
        if duration <= 0:
            continue

        subtitle_chunk_clips = create_subtitles(text, duration, clip_relative_start)
        subtitle_clips.extend(subtitle_chunk_clips)
        print(f"subtitle_clips: {subtitle_clips}")

    final_clip = CompositeVideoClip(
                [processed_clip.with_position('center')] + subtitle_clips,
                size=processed_clip.size
            )
                
    final_clip.audio = clip.audio

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

        

    def change_video_resolution(video_path):
        output_path = "./original_converted_ffmpeg_quality_test.mp4"
        (
        ffmpeg
        .input(video_path)
        .output(output_path, vf='scale=7680:4320')
        .run(overwrite_output=True)
        )
        return output_path

    from moviepy.video.fx.MultiplyColor import MultiplyColor  
    from moviepy.video.fx.FadeIn import FadeIn
    from moviepy.video.fx.FadeOut import FadeOut
    final_clip = MultiplyColor(factor=0.3).apply(final_clip) 
    final_clip = FadeIn(duration=1.0).apply(final_clip)
    final_clip = FadeOut(duration=1.0).apply(final_clip)

    print(f"video original fps: {clip.fps}")
    output_dir = "./Logging_and_filepaths/Video_clips"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{video_name}.mp4")
    final_clip.write_videofile(
        out_path,
        codec=video_codec or "libx264",
        audio_codec=audio_codec or "aac",
        bitrate=str(bitrate) or "4000k",
        preset="slow",
        threads=6,
        fps=clip.fps,
    )

    original_video = change_video_resolution(full_video)

    #test video quality
    test_videoquality_comparison(final_clip,original_video)

    print(f"video is completed: output path : {out_path}, video name: {video_name} video_fps: {clip.fps}, codec: {video_codec}, bitrate: {bitrate}, audio_codec: {audio_codec}, subtitles: {subtitles}")
    full_video.close()
    clip.close()






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
    
               try:
                  print("creating video now")
    
                  create_short_video(video_path=text_video_path, start_time=text_video_start_time, end_time = text_video_endtime, video_name = text_video_title,subtitle_text=subtitles)
               except Exception as e:
                  print(f"error during creation of video : {str(e)}")
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
            chunk = chunk_limiter.forward(file_path=transcripts_path, max_chars=2500)
               
        except Exception as e:
                print(f"Error during chunking from file {transcripts_path}: {e}")
                break

        if not chunk.strip():
                print("Finished processing current transcript. Now exiting func [Transcript Reasoning Agent]")
                Verify_Agent(Chunk_saving_text_file)
                break

        task = f"""
           You are a human-like reader analyzing & reading the chunk to decide if it contains motivational, inspirational, wisdom-based, or life-changing quotes or advice.
            You may find multiple motivational or inspirational quotes within the text chunk. Your task is to carefully analyze the entire chunk and:

            - Identify all separate quotes or pieces of wisdom worth saving.
            - Ignore non-motivational or normal talk that doesn’t meet the criteria.
            - For each valid quote, call SaveMotivationalQuote separately with the full timestamp and text.
            - Continue scanning the chunk to find additional quotes, even if normal talk appears between quotes.
            - Do NOT stop after finding the first quote.

            Example usage for saving multiple quotes in one chunk:

            SaveMotivationalQuote(text="[10.00s - 15.00s] Quote one text here.", text_file=text_file)
            SaveMotivationalQuote(text="[16.00s - 20.00s] Quote two text here.", text_file=text_file)

            Once all quotes are identified and saved, call final_answer("please provide me with next text to analyze").

            Look specifically for quotes or advice that:
            - Inspire action or courage
            - Share deep life lessons or universal truths
            - Teach about discipline, power, respect, or success
            - Offer practical wisdom or mindset shifts that can change how someone lives
            - Are emotionally uplifting or provoke reflection

            NOTE: One line in the chunk might not provide full context, but multiple lines in a chunk can provide full context & valuable quote to be saved, so consider reasoning over the entire chunk when answering.
            Often, several consecutive lines together (2 or 3 lines) may form a meaningful and powerful quote or insight worth saving, even if a single line alone does not.


            If you find such a quote & advice, use the `SaveMotivationalQuote` tool and include the timestamp of the quote.
            Here is an example: SaveMotivationalQuote(text="[3567.33s - 3569.65s] - The magic you are looking for is in the work you are avoiding.", text_file=text_file)

            Then proceed with the next chunk by using the `final_answer` tool if no more text is worth saving in the chunk.

            You don't need or are allowed to use any other tools than `SaveMotivationalQuote` and `final_answer`.

            Here is the chunk you will analyze using only reasoning like a human:

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
                chunk = chunk_limiter.forward(file_path=transcript_path)
                print(f"chunk for verify agent: {chunk} from {transcript_path}")
            except Exception as e:
                print(f"Error during chunking from file {transcript_path}: {e}")
                break

            if not chunk.strip():
                print("Finished processing current transcript.")
                set_current_videourl("")
                break

            task = f"""
            You are a human-like reader analyzing **one saved quote at a time** from another agent, who already deemed it motivational. Your task is to carefully verify whether this quote contains truly motivational, inspirational, wisdom-based, or life-changing advice suitable for a standalone motivational short video.

            Look specifically for quotes or advice that:  
            - Inspire action or courage  
            - Share deep life lessons or universal truths  
            - Teach about discipline, power, respect, or success  
            - Offer practical wisdom or mindset shifts that can change how someone lives  
            - Are emotionally uplifting or provoke reflection  
            - Provide full context and understanding  
            - can be used as a standalone motivational short video because it provide sutch motivational quote

            If you find the quote meets these criteria, use the `SaveMotivationalQuote_CreateShort` tool **including the original timestamp(s) exactly as they appear**.

            **Note:**  
            - If the quote spans multiple lines or segments, and each segment has its own timestamp, **include all timestamps exactly as saved by the first agent** when saving the quote.

            For example:  
            SaveMotivationalQuote_CreateShort(quote="[2323.0s - 2325.0s] Every great achievement begins with the decision to try. [2325.0s - 2327.0s] Courage doesn't always roar; sometimes it's the quiet voice at day's end saying 'I will try again tomorrow.'", text_file=text_file)

            Then proceed with the next quote/chunk by using `final_answer`.

            You may only use these tools:  
            - `SaveMotivationalQuote_CreateShort`  
            - `final_answer`

            Here is how to use them:  
            - SaveMotivationalQuote_CreateShort(text="New text saved:[858.98s - 866.98s] The magic you are looking for is in the work you are avoiding [866.98s - 875.00s] the only reason you are not living the life you want to live is because you [875.00s - 900.00s] day by day keep feeding the life you dont want to live", text_file=text_file)  
            - final_answer("please provide me with next text to analyze")

            Important:  
            - The quote you analyze is already somewhat motivational, but you must decide if it’s **good enough for a motivational short video**.  
            - **Preserve all timestamps exactly as saved by the first agent.**

            Here is the quote/chunk you will analyze using human-like reasoning:  

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
            model_id=r'c:\Users\didri\Desktop\LLM-models\Qwen2.5-7B-Instruct',
            device_map="auto",
            load_in_4bit=True,
            torch_dtype=torch.float16,
            max_new_tokens=1500,
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






if __name__ == "__main__":


    # print_gpu_stats() 
    # print(torch.cuda.get_device_name(0)) 
    # print(torch.cuda.get_device_name(1)) 
    # print(torch.cuda.is_available())    
    # try:
    #   video_paths = [
    #      r"c:\Users\didri\AppData\Local\CapCut\Videos\input_video.mp4"
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

    torch.cuda.empty_cache()
    gc.collect()
    text = """"
    [0.00s - 1.02s] How can people better?
    """
    set_current_videourl(r"c:\Users\didri\AppData\Local\CapCut\Videos\input_video.mp4")

    SaveMotivationalQuote_CreateShort(text,Final_saving_text_file)