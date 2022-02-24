from __future__ import annotations

from typing import TypeVar

import pytorch_lightning as pl
from pytorch_lightning.loggers import LightningLoggerBase, LoggerCollection


T = TypeVar("T", bound=LightningLoggerBase)

def get_logger_by_type(trainer: pl.Trainer, logger_class: type[T]) -> T | None:
    if isinstance(trainer.logger, LoggerCollection):
        return next((logger for logger in trainer.logger._logger_iterable if isinstance(logger, logger_class)), None)
    elif isinstance(trainer.logger, logger_class):
        return trainer.logger
    else:
        return None
