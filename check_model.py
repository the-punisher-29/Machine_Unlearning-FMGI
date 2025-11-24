from code.models import SmallCIFAR10CNN
import torch

model = SmallCIFAR10CNN()
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")
if num_params <= 1000000:
    print("Model size is compliant (<= 1M).")
else:
    print("Model size is NOT compliant (> 1M).")
