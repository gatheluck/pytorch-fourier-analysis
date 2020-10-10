import os
import sys
import torch
import pytorch_lightning as pl
from typing import Union

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pytorch_fourier_analysis.augmentations.base import MixAugmentationBase
from pytorch_fourier_analysis.lit.logger import get_epoch_end_log
from pytorch_fourier_analysis.shared import calc_error


class ClassificationModel(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.modules.loss._Loss,
        mixaugment: Union[None, MixAugmentationBase],
        optimizer_class: torch.optim.Optimizer,
        scheculer_class: torch.optim.lr_scheduler._LRScheduler,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.mixaugment = mixaugment
        self.optimizer_class = optimizer_class
        self.scheduler_class = scheculer_class

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def configure_optimizers(self):
        # all we need here is just instantiation. Other options are already set by functools.partial beforehand.
        optimizer = self.optimizer_class(params=self.model.parameters())
        scheduler = self.scheduler_class(optimizer=optimizer)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, t = batch

        if self.mixaugment:  # apply mix augmentation
            loss, retdict = self.mixaugment(self.model, self.criterion, x, t)
        else:  # vanilla train
            output = self.model(x)
            loss = self.criterion(output, t)
            retdict = dict(x=x.detach(), output=output.detach(), loss=loss.detach())

        # calculate error and create log dict.
        err1, err5 = calc_error(retdict["output"], t.detach(), topk=(1, 5))
        log = dict(train_loss=retdict["loss"], train_err1=err1, train_err5=err5)
        return dict(loss=loss, log=log)  # need to return loss for backward.

    def validation_step(self, batch, batch_idx):
        x, t = batch

        output = self.model(x)
        loss = self.criterion(output, t)
        retdict = dict(x=x.detach(), output=output.detach(), loss=loss.detach())

        # calculate error and create log dict.
        err1, err5 = calc_error(retdict["output"], t.detach(), topk=(1, 5))
        log = dict(val_loss=retdict["loss"], val_err1=err1, val_err5=err5)
        return log  # no need to return loss.

    def training_epoch_end(self, outputs):
        log_dict = get_epoch_end_log(outputs)  # resolve nest of outputs.
        log_dict["step"] = self.current_epoch
        return dict(log=log_dict)

    def validation_epoch_end(self, outputs):
        log_dict = get_epoch_end_log(outputs)  # resolve nest of outputs.
        log_dict["step"] = self.current_epoch
        return dict(log=log_dict)
