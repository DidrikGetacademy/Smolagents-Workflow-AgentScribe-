import onnxruntime as ort
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.onnx
from basicsr.archs.swinir_arch import SwinIR
import os
import requests






onnx_model_path = r"c:\Users\didri\Desktop\LLM-models\Video-upscale-models\realesrgan_x2plus.onnx"

print("Available providers:", ort.get_available_providers())


providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  
available_providers = [p for p in providers if p in ort.get_available_providers()]
print("Using providers:", available_providers)
session = ort.InferenceSession(onnx_model_path,provider=available_providers)

input_name = session.get_inputs()[0].name
input_meta = session.get_inputs()[0]
input_name = input_meta.name
input_type = input_meta.type
input_shape = input_meta.shape

print(f"Model expects input '{input_name}' with type '{input_type}' and shape {input_shape}")


if 'float16' in input_type:
    np_dtype = np.float16
elif 'float32' in input_type:
    np_dtype = np.float32
else:
    raise TypeError(f"Unexpected model input type: {input_type}")

# Load your input image
img_path = r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\test.png"
img = Image.open(img_path).convert('RGB')
w, h = img.size
new_w = (w // 4) * 4
new_h = (h // 4) * 4


img = ImageOps.pad(img, (new_w, new_h), method=Image.BICUBIC, color=(0, 0, 0))


img_np = np.array(img).astype(np.dtype) / 255.0


img_np = np.transpose(img_np, (2, 0, 1))[None, :, :, :]  


outputs = session.run(None, {input_name: img_np})

output_np = outputs[0]
output_img = np.clip(output_np[0].transpose(1, 2, 0), 0, 1) * 255
output_img = output_img.astype(np.uint8)

output_pil = Image.fromarray(output_img)
output_pil.save("output_upscaled.png")

