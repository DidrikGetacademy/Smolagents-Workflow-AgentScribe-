import torch
import gc

# Function to get GPU memory usage in MB
def clean_get_gpu_memory():
    return torch.cuda.memory_allocated() / (1024 ** 2)

print(f"Memory before cleaning: {clean_get_gpu_memory():.2f} MB")

torch.cuda.empty_cache()
gc.collect()

print(f"Memory after cleaning: {clean_get_gpu_memory():.2f} MB")
print("cleaned")

if __name__ == "__main__": 
    clean_get_gpu_memory()