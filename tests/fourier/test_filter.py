import os
import sys

import torch
import torchvision
from PIL import Image 

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pytorch_fourier_analysis import fourier


if __name__ == "__main__":
    x = Image.open("tests/testdata/blackswan.png")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor()
    ])
    x = transform(x).unsqueeze(0)  # (B,C,H,W)
    # x = torch.ones((3, 32, 32), dtype=torch.float)
    x = fourier.bandpass_filter(x, bandwidth=10, filter_mode="high_pass")
    torchvision.utils.save_image(x, "logs/bandpass.png")
