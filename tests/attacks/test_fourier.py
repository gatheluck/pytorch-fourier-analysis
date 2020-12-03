import os
import sys
import math

import torch
import torchvision

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pytorch_fourier_analysis.attacks import FourierAttack, logits_to_index
from pytorch_fourier_analysis import shared


class TestLogitsToIndex:
    def test_obvious_input(self):
        batch_size = 8
        s = 4
        noise = 0.01
        index_h = torch.randint(0, s, (batch_size,))
        index_w = torch.randint(0, s, (batch_size,))
        index_h_onehot = torch.nn.functional.one_hot(index_h, num_classes=s).float()  # (B,S)
        index_w_onehot = torch.nn.functional.one_hot(index_w, num_classes=s).float()  # (B,S)

        index_ans = torch.matmul(index_h_onehot.unsqueeze(-1), index_w_onehot.unsqueeze(-2))
        assert logits_to_index(index_h_onehot + noise, index_w_onehot + noise, scale_logits=10.).equal(index_ans)

    # def test_grad(self):
    #     b = 8
    #     s = 4

    #     index_h = torch.randint(0, s, (b,))
    #     index_w = torch.randint(0, s, (b,))
    #     logits_h = torch.nn.functional.one_hot(index_h, num_classes=s)  # (B,S)
    #     logits_w = torch.nn.functional.one_hot(index_w, num_classes=s)  # (B,S)

    #     logits = torch.cat([logits_h.unsqueeze(1), logits_w.unsqueeze(1)], dim=1).float() + 0.01  # (B,2,S)
    #     logits = logits.requires_grad_()

    #     index = logits_to_index(logits * 5, tau=1)  # (B,2)
    #     pseudo_loss = index.sum()
    #     pseudo_loss.backward()

    #     assert logits.grad is not None
