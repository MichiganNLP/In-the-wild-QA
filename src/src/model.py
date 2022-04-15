from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Mapping

import pytorch_lightning as pl
import torch
from overrides import overrides
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn as nn
from torchmetrics import BLEUScore, Metric, SQuAD
from torchmetrics.text import ROUGEScore

from src.metrics import IouF1, Perplexity, normalize_answer

TYPE_SPLIT = Literal["train", "val", "test"]
TYPE_BATCH = Mapping[str, torch.Tensor]


class AnswerWithEvidenceModule(pl.LightningModule, ABC):
    def __init__(self, **kwargs) -> None:  # noqa
        super().__init__()

        self.save_hyperparameters()

        self.answers_generation_enabled = False
        self.evidence_selection_enabled = False

        self.answer_metrics: Mapping[str, Metric] = nn.ModuleDict({"bleu1": BLEUScore(1), "bleu2": BLEUScore(2),
                                                                   "bleu3": BLEUScore(3)})
        self.rouge = ROUGEScore()
        # TODO: uncomment when torchmetrics releases this fix: https://github.com/PyTorchLightning/metrics/pull/912
        # self.bert_score = BERTScore(num_threads=0)
        self.perplexity = Perplexity()
        self.squad = SQuAD()

        self.iou_f1 = IouF1()

    @abstractmethod
    def _step(self, batch: TYPE_BATCH, split: TYPE_SPLIT) -> Mapping[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def _generative_step(self, batch: TYPE_BATCH, step_output: Mapping[str, torch.Tensor]) -> Mapping[str, Any]:
        raise NotImplementedError

    def _update_metrics(self, batch: TYPE_BATCH, step_output: Mapping[str, torch.Tensor],
                        generative_step_output: Mapping[str, Any], split: TYPE_SPLIT) -> None:
        if generated := generative_step_output.get("generated"):
            id_ = batch["id"]
            answers = batch["answers"]

            # We normalize the generated and the ground truth answers before computing the metrics.
            #
            # Note BLEU, ROUGE, and SQuAD metrics perform different normalizations by default, some of them can't be
            # changed. Still, it doesn't matter because they do basic stuff, as we do, that's good enough for
            # evaluation (similar for the tokenization). But we do something on our end because BLEU doesn't do
            # any normalization.
            normalized_generated = [normalize_answer(generated_instance) for generated_instance in generated]
            normalized_answers = [[normalize_answer(answer_instance) for answer_instance in answers_instance]
                                  for answers_instance in answers]

            for name, metric in self.answer_metrics.items():
                metric(normalized_generated, normalized_answers)
                self.log(f"{name}/{split}", metric)

            # BERTScore doesn't support multiple targets, and we have a variable number of answer.
            # We don't complicate it much and just evaluate the first answer (the original one). It's good enough.
            # first_normalized_answer = [normalized_answers_instance[0]
            #                            for normalized_answers_instance in normalized_answers]
            # self.bert_score(normalized_generated, first_normalized_answer)
            # self.log(f"bert_score_first_answer/{split}", self.bert_score)

            # We handle the following metrics manually by doing `update`, `compute` and `reset` because they return a
            # dictionary of tensors instead of a single tensor, so it can't be done automatically by PL.

            self.rouge.update(normalized_generated, normalized_answers)

            squad_format_generated = [{"prediction_text": generated_instance, "id": id_instance}
                                      for generated_instance, id_instance in zip(normalized_generated, id_)]
            squad_format_answers = [{"answers": {"text": answers_instance}, "id": id_instance}
                                    for answers_instance, id_instance in zip(normalized_answers, id_)]
            self.squad.update(squad_format_generated, squad_format_answers)

        if pred_spans := generative_step_output.get("pred_spans"):
            self.iou_f1(pred_spans, batch["evidence"].tolist())
            self.log(f"iou_f1/{split}", self.iou_f1)

    def _eval_step(self, batch: TYPE_BATCH, split: TYPE_SPLIT) -> None:
        step_output = self._step(batch, split=split)
        generative_step_output = self._generative_step(batch, step_output)
        self._update_metrics(batch, step_output, generative_step_output, split)

    @overrides(check_signature=False)
    def validation_step(self, batch: TYPE_BATCH, batch_idx) -> None:
        self._eval_step(batch, split="val")

    @overrides(check_signature=False)
    def test_step(self, batch: TYPE_BATCH, batch_idx) -> None:
        self._eval_step(batch, split="test")

    def _eval_epoch_end(self, split: TYPE_SPLIT) -> None:
        self.log_dict({f"{k}/{split}": v for k, v in self.rouge.compute().items()})
        self.rouge.reset()

        self.log_dict({f"{k}/{split}": v for k, v in self.squad.compute().items()})
        self.squad.reset()

    @overrides(check_signature=False)
    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._eval_epoch_end(split="val")

    @overrides(check_signature=False)
    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._eval_epoch_end(split="test")
