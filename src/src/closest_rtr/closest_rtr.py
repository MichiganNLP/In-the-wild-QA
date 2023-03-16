import math
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import sentence_transformers
import torch
from overrides import overrides
from rich.progress import track
from torch import nn

from src.model import AnswerWithEvidenceModule, Batch
from src.utils import iter_utils


class ClosestAnswerWithEvidenceModule(AnswerWithEvidenceModule):
    answers_generation_enabled = True

    def __init__(self, train_instances: Sequence[Mapping[str, Any]], embedding_model: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.train_instances = train_instances
        self.embedding_model = sentence_transformers.SentenceTransformer(embedding_model)

        self.embedding_model_kwargs = {"convert_to_tensor": True, "normalize_embeddings": True}

        # We encode just one question to allocate the size correctly.
        encoded_question = self.embedding_model.encode(train_instances[0]["question"], **self.embedding_model_kwargs)
        self.train_embeddings = nn.Parameter(torch.empty(len(self.train_instances), encoded_question.shape[-1]),
                                             requires_grad=False)

    def _on_eval_start(self) -> None:
        self.embedding_model_kwargs["device"] = self.device
        # We compute them here and not in `__init__` because here the model is already in the device.
        batch_size = 32
        train_embeddings = torch.cat([self.embedding_model.encode([instance["question"]
                                                                   for instance in instances
                                                                   if instance],
                                                                  **self.embedding_model_kwargs)
                                      for instances in track(iter_utils.batch_iter(self.train_instances, batch_size,
                                                                                   incomplete="fill"),
                                                         total=math.ceil(len(self.train_instances) / batch_size),
                                                         description="Encoding the training questions", transient=True,
                                                         disable=not self.trainer.is_global_zero)])
        self.train_embeddings.copy_(train_embeddings)

    @overrides
    def on_validation_start(self) -> None:
        self._on_eval_start()

    @overrides
    def on_test_start(self) -> None:
        self._on_eval_start()

    @overrides
    def _generative_step(self, batch: Batch, step_output: MutableMapping[str, torch.Tensor]) -> Mapping[str, Any]:
        test_embeddings = self.embedding_model.encode(batch["question"], **self.embedding_model_kwargs)
        similarity_scores = test_embeddings @ self.train_embeddings.T
        most_similar_ids = similarity_scores.argmax(dim=-1)
        return {"generated": [self.train_instances[most_similar_id.item()]["answers"][0]
                              for most_similar_id in most_similar_ids]}
