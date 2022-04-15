from collections import Counter
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import torch
from overrides import overrides

from src.model import AnswerWithEvidenceModule, TYPE_BATCH


class MostCommonAnswerWithEvidenceModule(AnswerWithEvidenceModule):
    answers_generation_enabled = True

    def __init__(self, train_instances: Sequence[Mapping[str, Any]], **kwargs) -> None:
        super().__init__(**kwargs)
        self.most_common_answer = Counter(instance["answers"][0] for instance in train_instances).most_common(n=1)[0][0]

    @overrides
    def _generative_step(self, batch: TYPE_BATCH, step_output: MutableMapping[str, torch.Tensor]) -> Mapping[str, Any]:
        return {"generated": [self.most_common_answer] * len(batch["question"])}
