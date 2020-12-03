import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pytorch_fourier_analysis.attacks import logits_to_index


class TestLogitsToIndex:
    def test_obvious_input(self):
        batch_size = 8
        s = 4
        noise = 0.01
        index_h = torch.randint(0, s, (batch_size,))
        index_w = torch.randint(0, s, (batch_size,))
        index_h_onehot = torch.nn.functional.one_hot(
            index_h, num_classes=s
        ).float()  # (B,H)
        index_w_onehot = torch.nn.functional.one_hot(
            index_w, num_classes=s
        ).float()  # (B,W)

        index_ans = torch.matmul(
            index_h_onehot.unsqueeze(-1), index_w_onehot.unsqueeze(-2)
        )
        assert logits_to_index(
            index_h_onehot + noise, index_w_onehot + noise, scale_logits=10.0
        ).equal(index_ans)

    def test_grad(self):
        batch_size = 8
        s = 4
        noise = 0.01
        index_h = torch.randint(0, s, (batch_size,))
        index_w = torch.randint(0, s, (batch_size,))
        logits_h = (
            torch.nn.functional.one_hot(index_h, num_classes=s).float() + noise
        )  # (B,H)
        logits_w = (
            torch.nn.functional.one_hot(index_w, num_classes=s).float() + noise
        )  # (B,W)
        logits_h.requires_grad_()
        logits_w.requires_grad_()

        index = logits_to_index(logits_h, logits_w, scale_logits=10.0)

        pseudo_loss = index.sum()
        pseudo_loss.backward()

        assert logits_h.grad is not None
        assert logits_w.grad is not None
