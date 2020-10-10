import omegaconf
import itertools
import torch
from torch import Tensor
import pytorch_lightning as pl
from typing import Dict, List, Union


def get_loggers(
    cfg: omegaconf.DictConfig, online_logger_api_key: str
) -> List[pl.loggers.base.LightningLoggerBase]:
    """
    Return list of logger used by pytorch lightning
    """
    # offline logger
    loggers = [
        pl.loggers.mlflow.MLFlowLogger(experiment_name="mlflow_output", tags=None)
    ]

    # online logger
    if online_logger_api_key:
        comet_logger = pl.loggers.CometLogger(
            api_key=online_logger_api_key, project_name=cfg.project_name
        )
        comet_logger.experiment.add_tags(cfg_to_tags(cfg))
        loggers.append(comet_logger)

    return loggers


def cfg_to_tags(cfg: omegaconf.DictConfig) -> List[str]:
    """
    Convert omegaconf to list.
    """
    if not cfg:
        return list()

    tags = list()
    for k, v in omegaconf.OmegaConf.to_container(cfg).items():
        if not isinstance(v, dict):
            tags.append(":".join([k, str(v)]))
        else:
            if "name" in v.keys():
                tags.append(":".join([k, v["name"]]))
            else:
                pass

    return tags


def get_epoch_end_log(
    outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
) -> Dict[str, Tensor]:
    """
    Fill the gap between single theread and data.parallel.
    Form of outputs is List[Dict[str, Tensor]] or List[List[Dict[str, Tensor]]]

    Args
        output: Output list from training step.
    """
    log = dict()

    # if list is nested, flatten them.
    if type(outputs[0]) is list:
        outputs = [x for x in itertools.chain(*outputs)]

    if "log" in outputs[0].keys():
        for key in outputs[0]["log"].keys():
            val = torch.stack([x["log"][key] for x in outputs]).mean().cpu()  # .item()
            log[key + "_avg"] = val
    else:
        for key in outputs[0].keys():
            val = torch.stack([x[key] for x in outputs]).mean().cpu()  # .item()
            log[key + "_avg"] = val

    return log
