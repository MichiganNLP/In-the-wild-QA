from __future__ import annotations

import inspect
from collections.abc import Mapping, MutableMapping
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from overrides import overrides
from torch.optim import AdamW
from transformers import PreTrainedTokenizerBase, T5ForConditionalGeneration, get_linear_schedule_with_warmup

from src.decoding import compute_answer_prob, compute_answer_probs
from src.metrics import Perplexity, get_best_evidence_spans
from src.model import AnswerWithEvidenceModule, TYPE_BATCH, TYPE_SPLIT
from src.transformer_models.clip_decoder import ClipWithDecoder
from src.transformer_models.t5_and_visual import T5AndVisual, T5AndVisualEvidence, T5AndVisualEvidenceIO, T5MultiTask


def log_lr(pl_module: pl.LightningModule, **kwargs) -> None:
    for i, optimizer in enumerate(pl_module.trainer.optimizers):
        for j, param_group in enumerate(optimizer.param_groups):
            if (lr := param_group.get("lr")) is not None:
                pl_module.log(f"lr_{i}_group_{j}", lr, **kwargs)


def model_type_to_class(model_type: str) -> type[T5ForConditionalGeneration]:  # noqa
    return {
        "t5_text_and_visual": T5AndVisual,
        "t5_evidence": T5AndVisualEvidence,
        "t5_evidence_io": T5AndVisualEvidenceIO,
        "t5_multi_task": T5MultiTask,
        "t5_train": T5ForConditionalGeneration,
        "t5_zero_shot": T5ForConditionalGeneration,
        "clip_decoder": ClipWithDecoder,
    }[model_type]


class TransformersAnswerWithEvidenceModule(AnswerWithEvidenceModule):
    def __init__(self, decoder_tokenizer: PreTrainedTokenizerBase, generate_kwargs: Mapping[str, Any] | None = None,
                 **kwargs) -> None:  # noqa
        super().__init__(**kwargs)

        self.decoder_tokenizer = decoder_tokenizer

        model_class = model_type_to_class(self.hparams.model_type)
        model_kwargs = {}
        if "visual_size" in inspect.signature(model_class.__init__).parameters:
            model_kwargs["visual_size"] = self.hparams.visual_size
        if self.hparams.model_type == "clip_decoder":
            model_kwargs["pretrained_clip_ckpt_path"] = self.hparams.pretrained_clip_ckpt_path
        self.model = model_class.from_pretrained(self.hparams.model_name_or_path, **model_kwargs)

        self.answers_generation_enabled = isinstance(self.model, T5ForConditionalGeneration)
        self.evidence_selection_enabled = "evidence" in self.hparams.model_type or "multi" in self.hparams.model_type

        self.visual_input_enabled = ("visual" in self.hparams.model_type
                                     or self.hparams.model_type.startswith("clip_")
                                     or self.evidence_selection_enabled)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.perplexity = Perplexity()

        self.generate_kwargs = generate_kwargs or {}

        self.generate_kwargs.setdefault("return_dict_in_generate", True)
        self.generate_kwargs.setdefault("output_scores", True)

        # The following are useful to compute the encoder layer output only once.
        self.generate_kwargs.setdefault("output_hidden_states", True)
        self.generate_kwargs.setdefault("output_attentions", True)

    @overrides(check_signature=False)
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                visual: torch.Tensor | None = None, visual_attention_mask: torch.Tensor | None = None,
                answer_ids: torch.Tensor | None = None, **kwargs) -> Any:
        if answer_ids is not None:
            answer_ids = answer_ids.clone()
            answer_ids[answer_ids == self.decoder_tokenizer.pad_token_id] = -100  # For the loss computation.

        if self.visual_input_enabled:
            kwargs["visual"] = visual
            kwargs["visual_attention_mask"] = visual_attention_mask

        if self.answers_generation_enabled:
            kwargs["labels"] = answer_ids

        return self.model(input_ids, attention_mask=attention_mask, **kwargs)

    def _evidence_loss(self, start_scores: torch.Tensor, end_scores: torch.Tensor, batch: TYPE_BATCH) -> torch.Tensor:
        evidence = batch["evidence"]
        starts, ends = evidence[..., 0], evidence[..., 1]

        # FIXME: we should ignore the padding frames.
        start_loss = self.cross_entropy_loss(start_scores, starts[:, 0])  # FIXME: we predict only one evidence.
        end_loss = self.cross_entropy_loss(end_scores, ends[:, 0])

        return start_loss + end_loss

    def _evidence_io_loss(self, visual_scores: torch.Tensor, batch: TYPE_BATCH) -> torch.Tensor:
        evidence = batch["evidence"]
        starts, ends = evidence[..., 0], evidence[..., 1]
        batch_size, evidence_number = starts.shape

        ground_truth = torch.zeros_like(visual_scores)
        for b_idx in range(batch_size):
            for e_idx in range(evidence_number):
                ground_truth[b_idx, starts[b_idx][e_idx]: ends[b_idx][e_idx]] = 1

        # FIXME: is this fine? Cross entropy supposes exactly one class is true, which is not the case here (classes are
        #  the 2nd index). Also, `ground_truth` should be a probability on that index
        #
        # FIXME: we should ignore the padding frames.
        return self.cross_entropy_loss(visual_scores, ground_truth)

    # Don't check the signature here because a transitive dependency has a bug when an argument has a `Literal` type
    # with a string. See https://github.com/bojiang/typing_utils/issues/10
    @overrides(check_signature=False)
    def _step(self, batch: TYPE_BATCH, split: TYPE_SPLIT) -> MutableMapping[str, torch.Tensor]:
        output = super()._step(batch, split)

        kwargs = {}

        if split != "train":
            kwargs["output_attentions"] = True
            kwargs["output_hidden_states"] = True

        model_output = self(input_ids=batch["question_ids"], attention_mask=batch["question_mask"],
                            visual=batch.get("visual"), visual_attention_mask=batch.get("visual_mask"),
                            answer_ids=batch["answer_ids"], **kwargs)

        if self.answers_generation_enabled and split != "train":
            output["encoder_hidden_states"] = model_output.encoder_hidden_states
            output["encoder_attentions"] = model_output.encoder_attentions

        if self.answers_generation_enabled:
            answer_loss = model_output.loss
            output["answer_logits"] = model_output.logits
        else:
            answer_loss = 0

        if self.evidence_selection_enabled:
            if self.hparams.model_type in {"t5_evidence", "t5_multi_task"}:
                if self.hparams.model_type == "t5_evidence":
                    start_scores, end_scores = model_output
                else:
                    start_scores, end_scores = model_output.start, model_output.end

                output["start_scores"] = start_scores
                output["end_scores"] = end_scores

                evidence_loss = self._evidence_loss(start_scores, end_scores, batch)
            elif self.hparams.model_type == "t5_evidence_io":
                output["visual_scores"] = model_output
                evidence_loss = self._evidence_io_loss(output["visual_scores"], batch)
            else:
                raise ValueError(f"Unknown model type: {self.hparams.model_type}")
        else:
            evidence_loss = 0

        output["loss"] = (getattr(self.hparams, "vqa_weight", 1) * answer_loss
                          + getattr(self.hparams, "evidence_weight", 1) * evidence_loss)
        self.log(f"loss/{split}", output["loss"], batch_size=len(batch["question"]))

        return output

    @overrides(check_signature=False)
    def training_step(self, batch: TYPE_BATCH, batch_idx: int = 0) -> torch.Tensor:
        loss = self._step(batch, split="train")["loss"]

        batch_size = len(batch["question"])
        self.log("batch_size", float(batch_size), batch_size=batch_size)

        log_lr(self, batch_size=batch_size)

        return loss

    @overrides
    def _generative_step(self, batch: TYPE_BATCH, step_output: MutableMapping[str, torch.Tensor]) -> Mapping[str, Any]:
        output = {}

        if self.answers_generation_enabled:
            kwargs = {}

            if self.visual_input_enabled:
                kwargs["visual"] = batch["visual"]
                kwargs["visual_attention_mask"] = batch["visual_mask"]

            # FIXME: this doesn't work with all models, because the encoders are different. We should call first the
            #  encoder and save the result. But for it to work, the models need to implement the encoder well.
            # if model_config.is_encoder_decoder:  # Reuse the encoder output to avoid computing it twice.
            #     kwargs["encoder_outputs"] = BaseModelOutput(last_hidden_state=step_output["encoder_hidden_states"][-1]
            #                                                 hidden_states=step_output["encoder_hidden_states"],
            #                                                 attentions=step_output["encoder_attentions"])

            generated_output = self.model.generate(batch["question_ids"], attention_mask=batch["question_mask"],
                                                   **self.generate_kwargs, **kwargs)
            output["generated_ids"] = generated_output.sequences
            output["generated"] = self.decoder_tokenizer.batch_decode(output["generated_ids"], skip_special_tokens=True)
            output["generated_scores"] = generated_output.scores

        if self.evidence_selection_enabled:
            if (visual_scores := step_output.get("visual_scores")) is None:
                start_scores, end_scores = step_output["start_scores"], step_output["end_scores"]
            else:
                # The start and end scores can be expressed in terms of the inside and outside scores. The score that
                # a span starts at a given moment is the moment inside score plus the outside score of the previous
                # moment. Similar reasoning for the end score.

                # The visual inside probability is binary, and we model it from sigmoid. The probability of outside (
                # the opposite) comes from the negation, which in sigmoid can be seen as negating the score as it's
                # symmetrical.
                neg_visual_scores = -visual_scores
                start_scores = torch.empty_like(visual_scores)
                start_scores[0] = visual_scores[0]
                start_scores[1:] = visual_scores[1:] + neg_visual_scores[:-1]

                end_scores = torch.empty_like(visual_scores)
                end_scores[-1] = visual_scores[-1]
                end_scores[:-1] = visual_scores[:-1] + neg_visual_scores[1:]

            start, end = get_best_evidence_spans(start_scores, end_scores, mask=batch["visual_mask"])
            # FIXME: the `end` should be summed 1 because it's going to be exclusive when measuring it.
            output["pred_spans"] = [list(zip(start_instance, end_instance))
                                    for start_instance, end_instance in zip(start.tolist(), end.tolist())]

        return output

    # Don't check the signature here because a transitive dependency has a bug when an argument has a `Literal` type
    # with a string. See https://github.com/bojiang/typing_utils/issues/10
    @overrides(check_signature=False)
    def _update_metrics(self, batch: TYPE_BATCH, step_output: MutableMapping[str, torch.Tensor],
                        generative_step_output: Mapping[str, Any], split: TYPE_SPLIT) -> None:
        super()._update_metrics(batch, step_output, generative_step_output, split)

        if (generated_ids := generative_step_output.get("generated_ids")) is not None:
            answer_ids = batch["answer_ids"]

            batch_size = len(answer_ids)

            model_config = self.model.config

            ground_truth_logits = step_output["answer_logits"]
            ground_truth_probs = compute_answer_probs(ground_truth_logits, answer_ids, model_config,
                                                      ignore_eos_token=True)
            ground_truth_prob = compute_answer_prob(ground_truth_probs)
            self.log(f"ground_truth_prob/{split}", ground_truth_prob, batch_size=batch_size)

            perplexity_mask = ((answer_ids != model_config.pad_token_id) & (answer_ids != model_config.eos_token_id))
            self.perplexity(ground_truth_probs, perplexity_mask)
            self.log(f"perplexity/{split}", self.perplexity, batch_size=batch_size)

            # Generate the answer and compute metrics based on it:

            generated_logits = torch.stack(generative_step_output["generated_scores"], dim=1)
            generated_probs = compute_answer_probs(generated_logits, generated_ids, model_config, ignore_eos_token=True)
            generated_prob = compute_answer_prob(generated_probs)
            self.log(f"generated_prob/{split}", generated_prob, batch_size=batch_size)

    @overrides
    def configure_optimizers(self) -> Mapping[str, Any]:
        no_decay = {"bias", "LayerNorm.weight"}
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=self.trainer.estimated_stepping_batches)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
