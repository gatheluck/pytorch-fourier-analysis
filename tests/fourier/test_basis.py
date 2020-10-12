import os
import sys
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pytorch_fourier_analysis import fourier


class TestBasis():
    def test_