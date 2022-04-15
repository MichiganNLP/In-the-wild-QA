import logging
import os

import pytorch_lightning as pl

from src.transformer_models.model import TransformersAnswerWithEvidenceModule

logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer: pl.Trainer, pl_module: TransformersAnswerWithEvidenceModule) -> None:
        logger.info("***** Validation results *****")
        if pl_module.should_log():
            for k, v in sorted(trainer.callback_metrics.items(), key=lambda x: x[0]):
                if k not in {"log", "progress_bar"}:
                    logger.info(f"{k} = {v}")

    def on_test_end(self, trainer: pl.Trainer, pl_module: TransformersAnswerWithEvidenceModule) -> None:
        logger.info("***** Test results *****")
        if pl_module.should_log():
            with open(os.path.join(pl_module.hparams.output_dir, "test_results.txt"), "w") as writer:
                for k, v in sorted(trainer.callback_metrics.items(), key=lambda x: x[0]):
                    if k not in {"log", "progress_bar"}:
                        message = f"{k} = {v}"
                        logger.info(message)
                        writer.write(f"{message}\n")
