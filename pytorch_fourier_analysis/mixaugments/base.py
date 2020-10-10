import torch


class MixAugmentationBase:
    def __init__(self):
        pass

    def _calc_mixed_loss(
        self,
        output: torch.tensor,
        t: torch.tensor,
        rand_index: torch.tensor,
        lam: float,
        criterion: torch.nn.modules.loss._Loss,
    ) -> torch.tensor:
        """
        Calcurate loss for mix augmentation.
        lamda * (loss for t_a)   + (1.0 - lamda) * (loss for t_b)

        Args
            output: Output logits from model. Shape should be [b, num_classes].
            t: Target classes. Shape should be [b].
            rand_index: Index of random swaping. Shape should be[b].
            lam: Weight of "a" side target. This value is same between same batch.
            criterion: Function which map from logits to loss.
        """
        t_a, t_b = t, t[rand_index]
        loss = (lam * criterion(output, t_a)) + (
            (1.0 - lam) * criterion(output, t_b)
        )
        return loss
