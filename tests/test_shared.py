import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pytorch_fourier_analysis.shared import get_model, calc_error


class TestGetModel:
    def test_valid_input(self):
        # test resnet50
        model = get_model("resnet50", num_classes=10)
        x = torch.randn(16, 3, 32, 32)
        assert model(x).size() == torch.Size([16, 10])

        # test wideresnet-40-2
        model = get_model("wideresnet40", num_classes=10)
        x = torch.randn(16, 3, 32, 32)
        assert model(x).size() == torch.Size([16, 10])


class TestClacError:
    def test_valid_input(self):
        # top-1 (all correct)
        output = torch.zeros(16, 10)
        output[:, 0] = 1.0
        target = torch.zeros(16)
        assert calc_error(output, target, topk=(1,))[0].equal(
            torch.Tensor([0.0]).float()
        )

        # top-1 (all wrong)
        output = torch.zeros(16, 10)
        output[:, 0] = 1.0
        target = torch.ones(16)
        assert calc_error(output, target, topk=(1,))[0].equal(
            torch.Tensor([100.0]).float()
        )

        # top-1 (all correct) and top-5 (all correct)
        output = torch.zeros(16, 10)
        output[:, 0] = 1.0
        output[:, 1] = 0.1
        target = torch.zeros(16)
        assert calc_error(output, target, topk=(1, 5))[0].equal(
            torch.Tensor([0.0]).float()
        )
        assert calc_error(output, target, topk=(1, 5))[1].equal(
            torch.Tensor([0.0]).float()
        )

        # top-1 (all wrong) and top-5 (all correct)
        output = torch.zeros(16, 10)
        output[:, 0] = 1.0
        output[:, 1] = 0.1
        target = torch.ones(16)
        assert calc_error(output, target, topk=(1, 5))[0].equal(
            torch.Tensor([100.0]).float()
        )
        assert calc_error(output, target, topk=(1, 5))[1].equal(
            torch.Tensor([0.0]).float()
        )
