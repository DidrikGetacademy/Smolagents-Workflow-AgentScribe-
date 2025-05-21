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
model_path = r""
train_file = os.path.join(model_path, "train.jsonl")
test_file = os.path.join(model_path, "test.jsonl")
output_dir = os.path.join(model_path, "")
os.makedirs(output_dir, exist_ok=True)



#https://medium.com/@danushidk507/qwen2-finetuning-qwen2-f89c5c9d15da

