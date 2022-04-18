from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping
from typing import Any, Literal

import pytorch_lightning as pl
import torch
from overrides import overrides
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn as nn
from torchmetrics import BLEUScore, Metric, SQuAD
from torchmetrics.text import BERTScore, ROUGEScore

from src.metrics import IouF1, normalize_answer
from src.utils.logger_utils import UnusedWeightsFilter

TYPE_SPLIT = Literal["train", "val", "test"]
TYPE_BATCH = Mapping[str, Any]


class AnswerWithEvidenceModule(pl.LightningModule, ABC):
    answers_generation_enabled = False
    evidence_selection_enabled = False

    def __init__(self, **kwargs) -> None:  # noqa
        super().__init__()

        self.save_hyperparameters()

        self.answer_metrics: Mapping[str, Metric] = nn.ModuleDict({"bleu1": BLEUScore(1), "bleu2": BLEUScore(2),
                                                                   "bleu3": BLEUScore(3)})
        self.rouge = ROUGEScore()
        self.bert_score = BERTScore(model_name_or_path="roberta-large", num_threads=0)
        self.squad = SQuAD()

        self.iou_f1 = IouF1()

    def _on_eval_start(self) -> None:
        self.bert_score.embedding_device = self.device

    @overrides
    def on_validation_start(self) -> None:
        self._on_eval_start()

    @overrides
    def on_test_start(self) -> None:
        self._on_eval_start()

    def _step(self, batch: TYPE_BATCH, split: TYPE_SPLIT) -> MutableMapping[str, torch.Tensor]:
        return {}

    @abstractmethod
    def _generative_step(self, batch: TYPE_BATCH, step_output: MutableMapping[str, torch.Tensor]) -> Mapping[str, Any]:
        raise NotImplementedError

    def _update_metrics(self, batch: TYPE_BATCH, step_output: MutableMapping[str, torch.Tensor],
                        generative_step_output: Mapping[str, Any], split: TYPE_SPLIT) -> None:
        batch_size = len(batch["question"])

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
                self.log(f"{name}/{split}", metric, batch_size=batch_size)

            # We handle the following metrics manually by doing `update`, `compute` and `reset` because they return a
            # dictionary of tensors instead of a single tensor, so it can't be done automatically by PL.

            self.rouge.update(normalized_generated, normalized_answers)

            squad_format_generated = [{"prediction_text": generated_instance, "id": id_instance}
                                      for generated_instance, id_instance in zip(normalized_generated, id_)]
            squad_format_answers = [{"answers": {"text": answers_instance}, "id": id_instance}
                                    for answers_instance, id_instance in zip(normalized_answers, id_)]
            self.squad.update(squad_format_generated, squad_format_answers)

            # BERTScore doesn't support multiple targets, and we have a variable number of answer.
            # We don't complicate it much and just evaluate the first answer (the original one). It's good enough.
            first_normalized_answer = [normalized_answers_instance[0]
                                       for normalized_answers_instance in normalized_answers]
            self.bert_score.update(normalized_generated, first_normalized_answer)

        if pred_spans := generative_step_output.get("pred_spans"):
            self.iou_f1(pred_spans, batch["evidence"].tolist())
            self.log(f"iou_f1/{split}", self.iou_f1, batch_size=batch_size)

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
        instance_count = sum(t.shape[0] for t in self.bert_score.preds_input_ids)

        if instance_count:
            self.log_dict({f"{k}/{split}": v for k, v in self.rouge.compute().items()}, batch_size=instance_count)
            self.rouge.reset()

            self.log_dict({f"{k}/{split}": v for k, v in self.squad.compute().items()}, batch_size=instance_count)
            self.squad.reset()

            unused_weights_filter = UnusedWeightsFilter()
            logging.getLogger("transformers.modeling_utils").addFilter(unused_weights_filter)

            self.log_dict({f"bert_score_first_answer_{k}/{split}": sum(v) / len(v)
                           for k, v in self.bert_score.compute().items()}, batch_size=instance_count)

            logging.getLogger("transformers.modeling_utils").removeFilter(unused_weights_filter)

            self.bert_score.reset()

    @overrides(check_signature=False)
    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._eval_epoch_end(split="val")

    @overrides(check_signature=False)
    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._eval_epoch_end(split="test")
