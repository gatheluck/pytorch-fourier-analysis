import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pytorch_fourier_analysis.attacks import logit_to_index, index_to_basis


class TestLogitToIndex:
    def test_obvious_input(self):
        batch_size = 8
        s = 4
        noise = 0.01
        index_h = torch.randint(0, s, (3 * batch_size,))
        index_w = torch.randint(0, s, (3 * batch_size,))
        index_h_onehot = (
            torch.nn.functional.one_hot(index_h, num_classes=s)
            .float()
            .view(batch_size, 3, s)
        )  # (B,3,H)
        index_w_onehot = (
            torch.nn.functional.one_hot(index_w, num_classes=s)
            .float()
            .view(batch_size, 3, s)
        )  # (B,3,W)

        index_ans = torch.matmul(
            index_h_onehot.unsqueeze(-1), index_w_onehot.unsqueeze(-2)
        )
        assert logit_to_index(
            index_h_onehot + noise, index_w_onehot + noise, scale_logit=10.0
        ).equal(index_ans)

    def test_grad(self):
        batch_size = 8
        s = 4
        noise = 0.01
        index_h = torch.randint(0, s, (3 * batch_size,))
        index_w = torch.randint(0, s, (3 * batch_size,))
        logit_h = (
            torch.nn.functional.one_hot(index_h, num_classes=s)
            .float()
            .view(batch_size, 3, s)
            + noise
        )  # (B,3,H)
        logit_w = (
            torch.nn.functional.one_hot(index_w, num_classes=s)
            .float()
            .view(batch_size, 3, s)
            + noise
        )  # (B,3,W)
        logit_h.requires_grad_()
        logit_w.requires_grad_()

        index = logit_to_index(logit_h, logit_w, scale_logit=10.0)

        pseudo_loss = index.sum()
        pseudo_loss.backward()

        assert logit_h.grad is not None
        assert logit_w.grad is not None


class TestIndexToBasis:
    def test_grad(self):
        index = torch.randn(8, 3, 32, 32)
        index.requires_grad_()

        basis = index_to_basis(index)

        pseudo_loss = basis.sum()
        pseudo_loss.backward()

        assert index.grad is not None
