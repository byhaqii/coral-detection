import torch
print("MPS available:", torch.backends.mps.is_available())
print("PyTorch:", torch.__version__)