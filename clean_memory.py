import torch
import gc


def clean_get_gpu_memory(threshold=0.8):
    allocated = torch.cuda.memory_allocated()
    max_allocated = torch.cuda.max_memory_allocated()
    if max_allocated == 0:
        usage_ratio = 0
    else:
        usage_ratio = allocated / max_allocated

    if usage_ratio > threshold:
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.ipc_collect()
        print(f"Cleaned GPU: Usage was {usage_ratio:.2f}")
    return allocated / (1024 ** 2) 

if __name__ == "__main__": 
    clean_get_gpu_memory()