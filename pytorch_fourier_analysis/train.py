import os
import sys
import logging
import functools

import hydra
import omegaconf
import torch
import pytorch_lightning as pl

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pytorch_fourier_analysis.shared as shared
import pytorch_fourier_analysis.lit
# from pytorch_fourier_analysis import shared, lit
from pytorch_fourier_analysis.lit import LitTrainerCallback, ClassificationModel


@hydra.main(config_path="conf/train.yaml")
def main(cfg: omegaconf.DictConfig) -> None:
    """
    Entry point function for training models.
    """
    # show config
    logging.info(cfg.pretty())

    # setup loggers
    api_key = os.environ.get("ONLINE_LOGGER_API_KEY")
    loggers = pytorch_fourier_analysis.lit.get_loggers(cfg, api_key)
    for logger in loggers:
        logger.log_hyperparams(omegaconf.OmegaConf.to_container(cfg))

    # setup checkpoint callback and trainer
    checkpoint_callback = pytorch_fourier_analysis.lit.get_checkpoint_callback(
        cfg.savedir, monitor=cfg.checkpoint_monitor, mode=cfg.checkpoint_mode
    )

    trainer = pl.Trainer(
        deterministic=False,
        benchmark=True,
        fast_dev_run=False,
        gpus=cfg.gpus,
        num_nodes=cfg.num_nodes,
        distributed_backend=cfg.distributed_backend,  # check https://pytorch-lightning.readthedocs.io/en/stable/trainer.html#distributed-backend
        max_epochs=cfg.epochs,
        min_epochs=cfg.epochs,
        logger=loggers,
        callbacks=[LitTrainerCallback()],
        checkpoint_callback=checkpoint_callback,
        default_root_dir=cfg.savedir,
        weights_save_path=cfg.savedir,
        resume_from_checkpoint=cfg.resume_ckpt_path
        if "resume_ckpt_path" in cfg.keys()
        else None,  # if not None, resume from checkpoint
    )

    # setup model
    model = shared.get_model(name=cfg.arch, num_classes=cfg.dataset.num_classes)

    # setup noise augmentation
    noiseaugment = shared.get_noiseaugment(cfg.noiseaugment)
    optional_transform = [noiseaugment] if noiseaugment else []

    # setup dataset
    train_transform = shared.get_transform(
        cfg.dataset.input_size,
        cfg.dataset.mean,
        cfg.dataset.std,
        train=True,
        optional_transform=optional_transform,
    )
    val_transform = shared.get_transform(
        cfg.dataset.input_size, cfg.dataset.mean, cfg.dataset.std, train=False
    )

    dataset_root = os.path.join(
        hydra.utils.get_original_cwd(), "data"
    )  # this is needed because hydra automatically change working directory.
    train_dataset_class = shared.get_dataset_class(
        cfg.dataset.name, root=dataset_root, train=True
    )
    val_dataset_class = shared.get_dataset_class(
        cfg.dataset.name, root=dataset_root, train=False
    )

    train_dataset = train_dataset_class(transform=train_transform)
    val_dataset = val_dataset_class(transform=val_transform)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    # make cosin_anealing lambda function. this is needed for manual cosin annealing.
    cosin_annealing_func = functools.partial(
        shared.cosin_annealing,
        total_steps=cfg.epochs * len(train_dataloader),
        lr_max=1.0,
        lr_min=1e-6 / cfg.optimizer.lr,
    )

    # setup optimizer
    optimizer_class = shared.get_optimizer_class(cfg.optimizer)
    scheduler_class = shared.get_scheduler_class(cfg.scheduler, cosin_annealing_func)

    # setup mix augmentation
    mixaugment = shared.get_mixaugment(cfg.mixaugment)

    # train
    criterion = torch.nn.CrossEntropyLoss()
    litmodel = ClassificationModel(
        model, criterion, mixaugment, optimizer_class, scheduler_class
    )
    trainer.fit(litmodel, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
