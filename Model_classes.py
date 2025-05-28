
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
import argparse
from concurrent.futures import ThreadPoolExecutor
from realesrgan import RealESRGANer
from SwinIR.models.network_swinir import SwinIR as net
from tqdm import tqdm
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import onnxruntime as ort
from ultralytics.utils.ops import non_max_suppression, xywh2xyxy
import pynvml
import torch.nn.functional as F  
from SwinIR.utils import util_calculate_psnr_ssim as util
import time
import logging
import numpy as np
import cv2
import torch
import onnxruntime as ort
from tqdm import tqdm
from collections import OrderedDict




Chunk_saving_text_file = r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Logging_and_filepaths\saved_transcript_storage.txt"
Final_saving_text_file=r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Logging_and_filepaths\final_saving_motivational.txt"
model_path_SwinIR_color_denoise15_pth = r"c:\Users\didri\Desktop\LLM-models\Video-upscale-models\SwinIR-M_noise15.pth"
model_path_SwinIR_color_denoise15_onnx = r"c:\Users\didri\Desktop\LLM-models\Video-upscale-models\SwinIR-M_noise15.onnx"
model_path_Swin_BSRGAN_X4_pth = r"c:\Users\didri\Desktop\LLM-models\Video-upscale-models\BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
model_path_Swin_BSRGAN_X4_onnx = r"c:\Users\didri\Desktop\LLM-models\Video-upscale-models\BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.onnx"
model_path_realesgran_x2_pth = r"c:\Users\didri\Desktop\LLM-models\Video-upscale-models\RealESRGAN_x2plus.pth"




class realesgran:
    def __init__(self, frames, device):
        self.model_path = r"c:\Users\didri\Desktop\LLM-models\Video-upscale-models\RealESRGAN_x2plus.pth"
        self.device = device
        self.frames_to_upscale = frames
        self.real_esrgan = None

    def load_model(self):
        self.real_esrgan = RealESRGANer(
            scale=2,
            model_path=self.model_path,
            model=None,  # Let RealESRGANer handle model loading
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
        for count, frame in enumerate(self.frames_to_upscale, 1):
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output, _ = self.real_esrgan.enhance(img_rgb, outscale=2)
            out_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            print(f"Upscaled frame {count}/{len(self.frames_to_upscale)}: {out_bgr.shape}")
            upscaled.append(out_bgr)

        return upscaled




class swinir_processor:
    def __init__(self, processed_frames, model_name):
        try:
            self.model_path = model_path_Swin_BSRGAN_X4_pth if "x4_GAN" in model_name else model_path_SwinIR_color_denoise15_pth
            self.model_name = model_name
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = None
            self.session = None
            self.use_onnx = self.model_path.endswith(".onnx")
            self.window_size = 8 
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
                    #    'TensorrtExecutionProvider',
                    #     'CUDAExecutionProvider',
                    #     'CPUExecutionProvider'
                providers = ['CUDAExecutionProvider','CPUExecutionProvider'] 
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
    
        return upscaled_frames
    
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