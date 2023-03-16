import random
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import torch
from overrides import overrides

from src.model import AnswerWithEvidenceModule, Batch


def _predict_random_span(duration: float) -> tuple[float, float]:
    t_1 = random.uniform(0, duration)
    t_2 = random.uniform(0, duration)

    if t_1 > t_2:
        t_1, t_2 = t_2, t_1

    return t_1, t_2


class RandomAnswerWithEvidenceModule(AnswerWithEvidenceModule):
    answers_generation_enabled = True
    evidence_selection_enabled = True

    def __init__(self, train_instances: Sequence[Mapping[str, Any]], span_prediction_count: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.train_answers = [instance["answer"] for instance in train_instances]
        self.span_prediction_count = span_prediction_count

    @overrides
    def _generative_step(self, batch: Batch, step_output: MutableMapping[str, torch.Tensor]) -> Mapping[str, Any]:
        return {"generated": random.choices(self.train_answers, k=len(batch["question"])),
                "pred_spans": [[_predict_random_span(duration_instance) for _ in range(self.span_prediction_count)]
                               for duration_instance in batch["duration"]]}
