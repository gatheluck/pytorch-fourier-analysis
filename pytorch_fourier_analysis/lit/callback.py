import os
import sys
import logging
import pytorch_lightning as pl

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pytorch_fourier_analysis import shared


def get_checkpoint_callback(
    savedir: str, monitor: str, mode: str
) -> pl.callbacks.ModelCheckpoint:
    r"""
    Args
        savedir: Checkpoint save dirctory.
        monitor: Quantity to monitor.
        mode: One of {auto, min, max}.
    """
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=os.path.join(savedir, "checkpoint", "{epoch}-{val_loss_avg:.2f}"),
        monitor=monitor,
        save_top_k=1,
        verbose=True,
        mode=mode,  # max or min. (specify which direction is improment for a monitor value.)
        save_weights_only=False,
    )
    return checkpoint_callback


class LitTrainerCallback(pl.callbacks.Callback):
    r"""
    Callback class used in [pytorch_lightning.trainer.Trainer] class.
    For detail, please check following docs:
    - https://pytorch-lightning.readthedocs.io/en/stable/callbacks.html
    - https://pytorch-lightning.readthedocs.io/en/stable/trainer.html#callbacks
    """

    def on_train_start(self, trainer, pl_module):
        logging.info("Training start.")

    def on_train_end(self, trainer, pl_module):
        logging.info("Training successfully ended.")

        # save state dict to local.
        local_save_path = os.path.join(
            trainer.weights_save_path, "model_weight_final.pth"
        )
        shared.save_model(
            trainer.model.module.model, local_save_path
        )  # trainer.model.module.model is model in LitModel class
        logging.info("Trained model is successfully saved to [%s]", local_save_path)

        # logging to online logger
        for logger in trainer.logger:
            if isinstance(logger, pl.loggers.comet.CometLogger):
                # log local log to comet: https://www.comet.ml/docs/python-sdk/Experiment/#experimentlog_asset_folder
                logger.experiment.log_asset_folder(
                    trainer.default_root_dir, log_file_name=True, recursive=True
                )

                # log model to comet: https://www.comet.ml/docs/python-sdk/Experiment/#experimentlog_model
                if trainer.model:
                    logger.experiment.log_model("checkpoint", local_save_path)
                    logging.info(
                        "Trained model is successfully saved to comet as state dict."
                    )
                else:
                    logging.info(
                        "There is no model to log because [trainer.model] is None."
                    )
