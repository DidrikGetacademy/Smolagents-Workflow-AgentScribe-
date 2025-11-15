import torch
import gc


import torch, gc

def clean_get_gpu_memory(threshold=0.8):
    allocated = torch.cuda.memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    usage_ratio = allocated / total

    if usage_ratio > threshold:
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.ipc_collect()
        print(f"Cleaned GPU: Usage was {usage_ratio:.2f} ({allocated / 1024**2:.1f} MB)")

    return allocated / (1024 ** 2)

if __name__ == "__main__":
    mem = clean_get_gpu_memory()
    print(f"Current GPU memory usage: {mem:.1f} MB")
