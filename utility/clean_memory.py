import torch
import gc


import torch, gc

import gc
import torch
import ctypes
import numpy as np
# from ..neon.log import log

def clean_get_gpu_memory(threshold=0.1):
    """Enhanced GPU memory cleanup with additional strategies"""

    # 1. Standard cleanup
    gc.collect()

    # 2. Aggressive CUDA cache clearing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
      #  torch.cuda.ipc_collect()  # Clean up IPC memory

        # 3. Reset peak memory stats (helps with fragmentation tracking)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

   # 4. Force Python heap compaction (aggressive)
    try:
        # Trim malloc() memory back to OS (Linux/Windows libc)
        if hasattr(ctypes.CDLL(None), 'malloc_trim'):
            ctypes.CDLL(None).malloc_trim(0)
    except Exception:
        pass

    # 5. NumPy memory cleanup
    np.empty((0,), dtype=np.uint8)  # Forces NumPy to release unused buffers

    # 6. Multiple GC passes for circular references
    for _ in range(3):
        gc.collect(generation=2)  # Full collection including generation 2

    # 7. Log memory state
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        mem_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e9
     #+   log(f"[clean_get_gpu_memory] Allocated: {mem_allocated:.2f}GB | Reserved: {mem_reserved:.2f}GB | Free: {mem_free:.2f}GB")


if __name__ == "__main__":
    # Example usage of clean_get_gpu_memory
    clean_get_gpu_memory(threshold=0.1)
