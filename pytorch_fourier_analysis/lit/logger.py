import omegaconf
import pytorch_lightning as pl
from typing import List


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
