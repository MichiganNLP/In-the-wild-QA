from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from overrides import overrides
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration, VisualBertForQuestionAnswering, \
    get_linear_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5EncoderModel, T5Stack

from src.transformer_models.t5_dataloader import get_dataset


def _combine_attention_masks(text_attention_mask: Optional[torch.Tensor] = None,
                             visual_attention_mask: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    if text_attention_mask is not None and visual_attention_mask is not None:
        text_batch_size = text_attention_mask.shape[0]
        visual_batch_size = visual_attention_mask.shape[0]
        beam_size = text_batch_size // visual_batch_size
        if beam_size > 1:
            visual_attention_mask = visual_attention_mask.repeat(beam_size, 1)
        return torch.cat([text_attention_mask, visual_attention_mask], dim=1)
    else:
        assert text_attention_mask is None and visual_attention_mask is None, \
            "Can't set the text or visual attention mask as one is empty and the other one isn't."
        return None


class TextVisualEncoder(T5Stack):  # noqa
    def __init__(self, t5stack: T5Stack, visual_size: int) -> None:
        super().__init__(t5stack.config, t5stack.embed_tokens)
        self.embed_video = nn.Linear(visual_size, self.embed_tokens.embedding_dim)

    @overrides(check_signature=False)
    def forward(self, text_token_ids: torch.Tensor, visual: torch.Tensor,  # noqa
                attention_mask: Optional[torch.Tensor] = None, visual_attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Union[BaseModelOutputWithPastAndCrossAttentions, tuple[torch.Tensor, ...]]:
        text_embedding = self.embed_tokens(text_token_ids)
        visual_embedding = self.embed_video(visual)
        embedding = torch.cat([text_embedding, visual_embedding], dim=1)
        attention_mask = _combine_attention_masks(attention_mask, visual_attention_mask)

        return super().forward(inputs_embeds=embedding, attention_mask=attention_mask, **kwargs)


class T5AndVisual(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, visual_size: int) -> None:
        super().__init__(config)
        self.encoder = TextVisualEncoder(self.encoder, visual_size)  # noqa

    @overrides(check_signature=False)
    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, visual: Optional[torch.Tensor] = None,
                                      visual_attention_mask: Optional[torch.Tensor] = None,
                                      **kwargs) -> Mapping[str, Any]:
        output = super().prepare_inputs_for_generation(input_ids, **kwargs)  # noqa
        output["visual"] = visual
        output["visual_attention_mask"] = visual_attention_mask
        return output

    @overrides(check_signature=False)
    def forward(self, masked_caption_ids: Optional[torch.Tensor] = None, visual: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None, visual_attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, **kwargs) -> Union[Seq2SeqLMOutput, tuple[torch.Tensor, ...]]:
        if "encoder_outputs" not in kwargs:
            kwargs["encoder_outputs"] = self.encoder(masked_caption_ids, visual=visual, attention_mask=attention_mask,
                                                     visual_attention_mask=visual_attention_mask, **kwargs)

        # The attention mask used in the encoder (the combined mask) is actually necessary for the decoder even when
        # providing "encoder_outputs". We could compute it once in the encoder, return it as one of its fields and use
        # it here. However, in `T5FillerModel.generative_step` we input the encoder outputs but without the mask
        # since it's constructed from the `generate` output which in turn only returns certain fields (not the mask).
        attention_mask = _combine_attention_masks(attention_mask, visual_attention_mask)

        return super().forward(attention_mask=attention_mask, labels=labels, **kwargs)  # noqa


class T5AndVisualEvidence(T5EncoderModel):  # noqa
    def __init__(self, config: T5Config, visual_size: int) -> None:
        super().__init__(config)
        self.encoder = TextVisualEncoder(self.encoder, visual_size)
        # NOTE: problems here!
        self.lgsm = nn.LogSoftmax(dim=2)
        self.start_vec = nn.Linear(self.config.d_model, 1)
        self.end_vec = nn.Linear(self.config.d_model, 1)

    @overrides(check_signature=False)
    def forward(self, masked_caption_ids: Optional[torch.Tensor] = None, visual: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None, visual_attention_mask: Optional[torch.Tensor] = None,
                evidence=None, evidence_mask=None, **kwargs) -> Union[Seq2SeqLMOutput, tuple[torch.Tensor, ...]]:
        outputs = self.encoder(masked_caption_ids, visual=visual, attention_mask=attention_mask,
                               visual_attention_mask=visual_attention_mask, **kwargs)
        # We only care about the last hidden state sequence
        batch_size, visual_start = masked_caption_ids.shape
        # _, visual_len, _ = visual.shape
        visual_hidden = outputs.last_hidden_state[:, visual_start:, :]

        # assume it is batch_size * visual_lang

        start = self.start_vec(visual_hidden)
        end = self.end_vec(visual_hidden)

        start = torch.transpose(start, 1, 2)  # batch_size * n = 1 * visual_len
        end = torch.transpose(end, 1, 2)  # batch_size * n = 1 * visual_len

        return start, end

    def predict(self, masked_caption_ids: Optional[torch.Tensor] = None, visual: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None, visual_attention_mask: Optional[torch.Tensor] = None,
                evidence=None, evidence_mask=None) -> Union[Seq2SeqLMOutput, tuple[torch.Tensor, ...]]:
        start, end = self.forward(masked_caption_ids, visual, attention_mask, visual_attention_mask, evidence,
                                  evidence_mask)

        # assume it is batch_size * visual_len

        start_lgsm = self.lgsm(start)
        end_lgsm = self.lgsm(end)

        return start_lgsm, end_lgsm


class FineTuner(pl.LightningModule):  # noqa
    def __init__(self, **kwargs) -> None:  # noqa
        super().__init__()

        self.save_hyperparameters()

        if self.hparams.model_type == "T5_text_and_visual":
            self.model = T5AndVisual.from_pretrained(self.hparams.model_name_or_path,
                                                     visual_size=self.hparams.visual_size)
        elif self.hparams.model_type == "T5_evidence":
            self.model = T5AndVisualEvidence.from_pretrained(self.hparams.model_name_or_path,
                                                             visual_size=self.hparams.visual_size)
            self.xentloss = nn.CrossEntropyLoss()
        elif self.hparams.model_type == "visual_bert_QA":
            self.model = VisualBertForQuestionAnswering.from_pretrained(self.hparams.model_name_or_path,
                                                                        visual_size=self.hparams.visual_size)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name_or_path)

    def forward(
            self, input_ids, attention_mask=None, visual=None, visual_attention_mask=None,
            decoder_input_ids=None, decoder_attention_mask=None, labels=None,
            evidence=None, evidence_mask=None
    ):
        if self.hparams.model_type in {"T5_text_and_visual", "visual_bert_QA"}:
            return self.model(
                input_ids,
                attention_mask=attention_mask,
                visual=visual,
                visual_attention_mask=visual_attention_mask,
                labels=labels,
            )
        elif self.hparams.model_type == "T5_evidence":
            return self.model(
                input_ids,
                attention_mask=attention_mask,
                visual=visual,
                visual_attention_mask=visual_attention_mask,
                evidence=evidence,
                evidence_mask=evidence_mask
            )
        else:
            return self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
            )

    def _step(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, float, float]:
        if self.hparams.model_type == "T5_evidence":
            raw_start, raw_end = self(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
                visual=batch["visual_ids"],
                visual_attention_mask=batch["visual_mask"]
            )

            evidence = batch["evidence"]
            evidence_mask = batch["evidence_mask"]  # shape: batch_size * n

            starts = evidence[:, :, 0].long()
            ends = evidence[:, :, 1].long()

            start_lss, end_lss = [], []

            batch_size, n = starts.shape
            for b in range(batch_size):
                # for i in range(n):
                # multiple evidences (n >= 1)

                if int(evidence_mask[b][0].item()):
                    # otherwise, we should not care about the cross entropy loss
                    start_lss.append(self.xentloss(raw_start[b], starts[b][0].unsqueeze(dim=0)))
                    end_lss.append(self.xentloss(raw_end[b], ends[b][0].unsqueeze(dim=0)))

            # output start_pos: batch_size * n (number of evidences within an instance)
            tt_start_lss = torch.sum(torch.stack(start_lss))
            tt_end_lss = torch.sum(torch.stack(end_lss))

            # following the BERT paper of using the average of start and end loss
            # no accuracy or sentence accuracy available here

            return tt_start_lss + tt_end_lss, 0, 0
        else:
            labels = batch["target_ids"]
            labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

            if self.hparams.model_type in ["T5_text_and_visual", "visual_bert_QA"]:
                outputs = self(
                    input_ids=batch["source_ids"],
                    attention_mask=batch["source_mask"],
                    visual=batch["visual_ids"],
                    visual_attention_mask=batch["visual_mask"],
                    labels=labels,
                    decoder_attention_mask=batch["target_mask"]
                )
            else:
                outputs = self(
                    input_ids=batch["source_ids"],
                    attention_mask=batch["source_mask"],
                    labels=labels,
                    decoder_attention_mask=batch["target_mask"]
                )

            loss = outputs[0]

            preds = torch.argmax(outputs[1].softmax(-1), dim=-1)
            acc = torch.eq(preds, labels).sum() / (labels != -100).sum()  # token level acc
            n, length = preds.shape
            sent_acc = []
            for i in np.arange(n):
                p = preds[i]
                t = labels[i]
                p[t[:] == -100] = -100
                sent_acc.append(torch.equal(p, t))
            sent_acc = sum(sent_acc) / len(sent_acc)
            return loss, acc, sent_acc

    def training_step(self, batch: Mapping[str, Any], batch_idx: int = 0) -> torch.Tensor:
        loss, acc, sent_acc = self._step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        if not self.hparams.model_type == "T5_evidence":
            self.log("train_acc", acc, on_step=True, on_epoch=True)
            self.log("train_sentence_acc", sent_acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Mapping[str, Any], batch_idx) -> None:
        loss, acc, sent_acc = self._step(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        if not self.hparams.model_type == "T5_evidence":
            self.log("val_acc", acc, on_step=True, on_epoch=True)
            self.log("val_sentence_acc", sent_acc, on_step=True, on_epoch=True)

    def configure_optimizers(self) -> Mapping[str, Any]:
        """Prepare optimizer and schedule (linear warmup and decay)"""

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

        t_total = (
                (len(self.train_dataloader()) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=t_total)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def train_dataloader(self) -> DataLoader:
        train_dataset = get_dataset(tokenizer=self.tokenizer, data_dir=self.hparams.train_data, args=self.hparams)
        return DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True,
                          num_workers=4, collate_fn=my_collate, pin_memory=True, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        val_dataset = get_dataset(tokenizer=self.tokenizer, data_dir=self.hparams.dev_data, args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, collate_fn=my_collate, num_workers=4,
                          pin_memory=True, persistent_workers=True)

    @staticmethod
    def should_log() -> bool:
        """ Ask logger to log """
        return True


def my_collate(examples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    # len(examples) == specified batch_size
    evidence = [ex["evidence"] for ex in examples]
    max_len = max(len(ev) for ev in evidence)
    if evidence[0]:
        # whether the evidence should be used for training
        evidence_mask = [torch.Tensor([1] * len(ev) + [0] * (max_len - len(ev))) for ev in evidence]
        evidence = [torch.Tensor(ev + [(0, 0)] * (max_len - len(ev))) for ev in evidence]
        return {
            "source_ids": torch.stack([ex["source_ids"] for ex in examples]),
            "source_mask": torch.stack([ex["source_mask"] for ex in examples]),
            "visual_ids": torch.stack([ex["visual_ids"] for ex in examples]),
            "visual_mask": torch.stack([ex["visual_mask"] for ex in examples]),
            "target_ids": torch.stack([ex["target_ids"] for ex in examples]),
            "target_mask": torch.stack([ex["target_mask"] for ex in examples]),
            "evidence": torch.stack(evidence),
            "evidence_mask": torch.stack(evidence_mask)
        }
    elif examples[0]["target_ids"].nelement():
        return {
            "source_ids": torch.stack([ex["source_ids"] for ex in examples]),
            "source_mask": torch.stack([ex["source_mask"] for ex in examples]),
            "visual_ids": torch.stack([ex["visual_ids"] for ex in examples]),
            "visual_mask": torch.stack([ex["visual_mask"] for ex in examples]),
            "target_ids": torch.stack([ex["target_ids"] for ex in examples]),
            "target_mask": torch.stack([ex["target_mask"] for ex in examples]),
            "evidence": torch.tensor([]),
            "evidence_mask": torch.tensor([])
        }
    else:
        return {
            "source_ids": torch.stack([ex["source_ids"] for ex in examples]),
            "source_mask": torch.stack([ex["source_mask"] for ex in examples]),
            "visual_ids": torch.stack([ex["visual_ids"] for ex in examples]),
            "visual_mask": torch.stack([ex["visual_mask"] for ex in examples]),
            "target_ids": torch.tensor([]),
            "target_mask": torch.tensor([]),
            "evidence": torch.tensor([]),
            "evidence_mask": torch.tensor([])
        }
