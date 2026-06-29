import torch
from models.sngp_hdd import get_model

# ---- Create model ----
model = get_model(num_classes=16)  # adjust to your dataset
model.eval()

# ---- Count parameters ----
num_params = sum(p.numel() for p in model.parameters())
num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {num_params/1e6:.2f} M")
print(f"Trainable parameters: {num_trainable/1e6:.2f} M")