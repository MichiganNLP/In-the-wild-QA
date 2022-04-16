from __future__ import annotations

import logging
import re

from typing import TypeVar

import pytorch_lightning as pl
from pytorch_lightning.loggers import LightningLoggerBase, LoggerCollection


T = TypeVar("T", bound=LightningLoggerBase)

def get_logger_by_type(trainer: pl.Trainer, logger_class: type[T]) -> T | None:
    if isinstance(trainer.logger, LoggerCollection):
        return next((logger for logger in trainer.loggers if isinstance(logger, logger_class)), None)
    elif isinstance(trainer.logger, logger_class):
        return trainer.logger
    else:
        return None


class UninitializedWeightsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno != logging.WARNING \
               or re.match(r"^Some weights of [\w-]+ were not initialized from the model .+",
                           record.getMessage()) is None


class UnusedWeightsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno != logging.WARNING \
               or re.match(r"^Some weights of the model checkpoint at [\w-]+ were not used when initializing .+",
                           record.getMessage()) is None
