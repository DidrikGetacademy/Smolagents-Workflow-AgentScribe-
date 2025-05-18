import os
import torch
import gc
gc.collect()
torch.cuda.empty_cache()
from unsloth import FastLanguageModel

max_seq_length = 2048


# -----------------------------
# Paths
# -----------------------------
model_path = r"C:/Users/didri/Desktop/Programmering/VideoEnchancer program/local_model/microsoft/microsoft/Phi-3-mini-128k-instruct"
train_file = os.path.join(model_path, "train.jsonl")
test_file = os.path.join(model_path, "test.jsonl")
output_dir = os.path.join(model_path, "phi_3_finetuned")
os.makedirs(output_dir, exist_ok=True)



#https://medium.com/@danushidk507/qwen2-finetuning-qwen2-f89c5c9d15da

