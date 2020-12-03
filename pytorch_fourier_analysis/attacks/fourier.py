import torch

from .attack import AttackWrapper


def logits_to_index(logits_h: torch.FloatTensor,
                    logits_w: torch.FloatTensor,
                    tau: float = 1.0,
                    scale_logits: float = 5.0) -> torch.FloatTensor:
    """
    convert logits to index by gumbel softmax.
    the shape of logits should be (B,S). here S = math.ceil(H/2) = math.ceil(W/2).
    the shape of return is (B,2).

    Args
        logits: logits of fourier basis. its shape should be (B,2,S).
    """
    # apply gumbel softmax with "straight-through" trick.
    index_h_onehot = torch.nn.functional.gumbel_softmax(logits_h * scale_logits, tau=tau, hard=True)
    index_w_onehot = torch.nn.functional.gumbel_softmax(logits_w * scale_logits, tau=tau, hard=True)

    return torch.matmul(index_h_onehot.unsqueeze(-1), index_w_onehot.unsqueeze(-2))  # (B,H,W)


class FourierAttack(AttackWrapper):
    def __init__(self, num_iteration: int):
        self.num_iteration = num_iteration

    def _forward(
        self, pixel_model: torch.nn.Module, pixel_x: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return perturbed input in pixel space [0,255]
        """

        # init delta (=fourier basis)

        # compute delta in pixel space
        if self.num_iteration:
            pass

    def _run(self,
             pixel_model: torch.nn.Module):
        pass
