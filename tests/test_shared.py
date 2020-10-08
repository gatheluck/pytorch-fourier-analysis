import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pytorch_fourier_analysis.shared import get_model


class TestGetModel:
    def test_valid_input(self):
        # test resnet50
        model = get_model("resnet50", num_classes=10)
        x = torch.randn(16, 3, 32, 32)
        assert model(x).size() == torch.Size([16, 10])

        # test wideresnet-40-2
        model = get_model("wideresnet40", num_classes=10, widening_factor=2)
        x = torch.randn(16, 3, 32, 32)
        assert model(x).size() == torch.Size([16, 10])
