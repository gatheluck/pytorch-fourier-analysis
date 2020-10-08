import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pytorch_fourier_analysis import models


class TestWideResNet:
    def test_valid_input(self):
        # test wideresnet 16
        model = models.wideresnet16(num_classes=10)
        x = torch.randn(16, 3, 32, 32)
        assert model(x).size() == torch.Size([16, 10])

        model = models.wideresnet16(num_classes=100, widening_factor=1)
        x = torch.randn(16, 3, 32, 32)
        assert model(x).size() == torch.Size([16, 100])

        model = models.wideresnet16(num_classes=100, widening_factor=10, droprate=0.5)
        x = torch.randn(16, 3, 32, 32)
        assert model(x).size() == torch.Size([16, 100])

        # test wideresnet 28
        model = models.wideresnet28(num_classes=10)
        x = torch.randn(16, 3, 32, 32)
        assert model(x).size() == torch.Size([16, 10])

        model = models.wideresnet28(num_classes=100, widening_factor=1)
        x = torch.randn(16, 3, 32, 32)
        assert model(x).size() == torch.Size([16, 100])

        model = models.wideresnet28(num_classes=100, widening_factor=10, droprate=0.5)
        x = torch.randn(16, 3, 32, 32)
        assert model(x).size() == torch.Size([16, 100])

        # test wideresnet 40
        model = models.wideresnet40(num_classes=10)
        x = torch.randn(16, 3, 32, 32)
        assert model(x).size() == torch.Size([16, 10])

        model = models.wideresnet40(num_classes=100, widening_factor=1)
        x = torch.randn(16, 3, 32, 32)
        assert model(x).size() == torch.Size([16, 100])

        model = models.wideresnet40(num_classes=100, widening_factor=10, droprate=0.5)
        x = torch.randn(16, 3, 32, 32)
        assert model(x).size() == torch.Size([16, 100])
