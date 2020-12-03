import torch
import torch.fft

from .attack import AttackWrapper


def logits_to_index(
    logits_h: torch.FloatTensor,
    logits_w: torch.FloatTensor,
    tau: float = 1.0,
    scale_logits: float = 5.0,
) -> torch.FloatTensor:
    """
    convert logits to index using gumbel softmax.
    the shape of returned index is (B,3,H,W).

    Args
        logits_h: logits of fourier basis about hight. its shape should be (B,3,H).
        logits_h: logits of fourier basis about width. its shape should be (B,3,W) or (B,3,W//2+1)).
        tau: tempalature of Gumbel softmax.
        scale_logits: scale factor of logits. NOTE: if this value is too small, Gumbel softmax does not work instead of thevalue of tau.
    """
    # apply gumbel softmax with "straight-through" trick.
    index_h_onehot = torch.nn.functional.gumbel_softmax(
        logits_h * scale_logits, tau=tau, hard=True
    )
    index_w_onehot = torch.nn.functional.gumbel_softmax(
        logits_w * scale_logits, tau=tau, hard=True
    )

    return torch.matmul(
        index_h_onehot.unsqueeze(-1), index_w_onehot.unsqueeze(-2)
    )  # (B,3,H,W)


def index_to_basis(index: torch.FloatTensor) -> torch.FloatTensor:
    """
    convert index to Fourier basis by 2D FFT.
    in order to apply 2D FFT, dim argument of torch.fft.irfftn should be =(-2,-1).

    Args
        index: its shape should be (B,3,H,W//2+1).
    """
    _, _, h, _ = index.size()
    return torch.fft.irfftn(index, s=(h, h), dim=(-2, -1))


class FourierAttack(AttackWrapper):
    def __init__(self, num_iteration: int):
        self.num_iteration = num_iteration

    def _forward(
        self,
        pixel_model: torch.nn.Module,
        pixel_x: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return perturbed input in pixel space [0,255]
        """

        # init delta (=fourier basis)

        # compute delta in pixel space
        if self.num_iteration:
            pass

    def _run(
        self,
        pixel_model: torch.nn.Module,
        logits_h: torch.FloatTensor,
        logits_w: torch.FloatTensor,
        target: torch.Tensor,
    ):
        pass

        for it in range(self.num_iteration):
            loss = self.criterion(logit, target)
            loss.backward()
