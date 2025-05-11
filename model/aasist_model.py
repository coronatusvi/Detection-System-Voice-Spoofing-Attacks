import torch
from model.aasist_l import Model

def load_aasist_model(path="checkpoints/best.pth"):
    d_args = {
        "first_conv": 128,
        "filts": [[70, 1], [1, 32], [32, 32], [32, 24], [24, 24]],
        "gat_dims": [24, 32],
        "pool_ratios": [0.4, 0.5, 0.7, 0.7],  # ✅ Đã fix: dùng cùng tỉ lệ cho hS1/hT1 và hS2/hT2
        "temperatures": [2.0, 2.0, 100.0, 100.0]
    }

    model = Model(d_args)
    checkpoint = torch.load(path, map_location="cpu")
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model
