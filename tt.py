import torch
print(torch.version)         # should show +cu128
print(torch.cuda.is_available())  # should be True
