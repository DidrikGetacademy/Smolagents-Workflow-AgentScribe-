import gc 
gc.collect()
import torch 
torch.cuda.empty_cache()
from basicsr.utils.registry import ARCH_REGISTRY


print(list(ARCH_REGISTRY.keys()))

ARCH_REGISTRY._obj_map.clear()
