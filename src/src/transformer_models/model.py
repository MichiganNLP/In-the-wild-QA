from typing import Any, Literal, Mapping, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from overrides import overrides
from torch.optim import AdamW
from transformers import PreTrainedTokenizerBase, T5Config, T5ForConditionalGeneration, \
    VisualBertForQuestionAnswering, get_linear_schedule_with_warmup
from transformers.models.t5.modeling_t5 import T5EncoderModel

from src.transformer_models.t5_and_visual import T5AndVisual, TextVisualEncoder


class T5AndVisualEvidence(T5EncoderModel):  # noqa
    def __init__(self, config: T5Config, visual_size: int) -> None:
        super().__init__(config)
        self.encoder = TextVisualEncoder(self.encoder, visual_size)
        self.start_end = nn.Linear(self.config.d_model, 2)

    @overrides(check_signature=False)
    def forward(self, masked_caption_ids: Optional[torch.Tensor] = None, visual: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None, visual_attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(masked_caption_ids, visual=visual, attention_mask=attention_mask,
                               visual_attention_mask=visual_attention_mask, **kwargs)

        visual_start = masked_caption_ids.shape[1]
        visual_hidden = outputs.last_hidden_state[:, visual_start:, :]

        start_end = self.start_end(visual_hidden)

        return start_end[..., 0], start_end[..., 1]

    def predict(self, masked_caption_ids: Optional[torch.Tensor] = None, visual: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                visual_attention_mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        start, end = self(masked_caption_ids, visual, attention_mask, visual_attention_mask)
        return start.log_softmax(dim=-1), end.log_softmax(dim=-1)


class T5AndVisualEvidenceIO(T5EncoderModel):  # noqa
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
                evidence=None, evidence_mask=None, **kwargs) -> Union[Seq2SeqLMOutput, Tuple[torch.Tensor, ...]]:
        outputs = self.encoder(masked_caption_ids, visual=visual, attention_mask=attention_mask,
                               visual_attention_mask=visual_attention_mask, **kwargs)
        # We only care about the last hidden state sequence
        batch_size, visual_start = masked_caption_ids.shape
        # _, visual_len, _ = visual.shape
        visual_hidden = outputs.last_hidden_state[:, visual_start:, :]

        # assume it is batch_size * visual_leng

        start = self.start_vec(visual_hidden)
        end = self.end_vec(visual_hidden)

        start = torch.transpose(start, 1, 2)  # batch_size * N = 1 * visual_len
        end = torch.transpose(end, 1, 2)  # batch_size * N = 1 * visual_len

        return start, end

    def predict(self, masked_caption_ids: Optional[torch.Tensor] = None, visual: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None, visual_attention_mask: Optional[torch.Tensor] = None,
                evidence=None, evidence_mask=None) -> Union[Seq2SeqLMOutput, Tuple[torch.Tensor, ...]]:
        start, end = self.forward(masked_caption_ids, visual, attention_mask, visual_attention_mask, evidence,
                                  evidence_mask)

        # assume it is batch_size * visual_len

        start_lgsm = self.lgsm(start)
        end_lgsm = self.lgsm(end)

        return start_lgsm, end_lgsm


class FineTuner(pl.LightningModule):  # noqa
    def __init__(self, tokenizer: PreTrainedTokenizerBase, **kwargs) -> None:  # noqa
        super().__init__()

        self.tokenizer = tokenizer

        self.save_hyperparameters()

        if self.hparams.model_type == "T5_text_and_visual":
            self.model = T5AndVisual.from_pretrained(self.hparams.model_name_or_path,
                                                     visual_size=self.hparams.visual_size)
        elif self.hparams.model_type == "T5_evidence":
            self.model = T5AndVisualEvidence.from_pretrained(self.hparams.model_name_or_path,
                                                             visual_size=self.hparams.visual_size)
<<<<<<< HEAD
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.tokenizer_name_or_path)
            self.xentloss = nn.CrossEntropyLoss()

        elif self.hparams.model_type == "T5_evidence_IO":
            self.model = T5AndVisualEvidenceIO.from_pretrained(self.hparams.model_name_or_path,
                                                                visual_size=self.hparams.visual_size)   
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.tokenizer_name_or_path)
            self.xentloss = nn.CrossEntropyLoss()
            
=======
            self.cross_entropy_loss = nn.CrossEntropyLoss()
        elif self.hparams.model_type == "visual_bert_QA":
            self.model = VisualBertForQuestionAnswering.from_pretrained(self.hparams.model_name_or_path,
                                                                        visual_size=self.hparams.visual_size)
>>>>>>> d78b61ca48a4d11c0b0b03f4654ef6203a5b774f
        else:
            assert self.hparams.model_type in ["T5_train", "T5_zero_shot"]
            self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)

    @overrides(check_signature=False)
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                visual: Optional[torch.Tensor] = None, visual_attention_mask: Optional[torch.Tensor] = None,
                decoder_input_ids: Optional[torch.Tensor] = None,
                decoder_attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> Any:
        if labels is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100  # For the loss computation.

        if self.hparams.model_type in {"T5_text_and_visual", "visual_bert_QA"}:
            return self.model(input_ids, attention_mask=attention_mask, visual=visual,
                              visual_attention_mask=visual_attention_mask, labels=labels)
        elif self.hparams.model_type == "T5_evidence":
            return self.model(input_ids, attention_mask=attention_mask, visual=visual,
                              visual_attention_mask=visual_attention_mask)
        else:
            return self.model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                              decoder_attention_mask=decoder_attention_mask, labels=labels)

    def _step(self, batch: Mapping[str, Any], split: Literal["train", "val", "test"]) -> Optional[torch.Tensor]:
        if self.hparams.model_type == "T5_evidence":
            raw_start, raw_end = self(input_ids=batch["source_ids"], attention_mask=batch["source_mask"],
                                      visual=batch["visual_ids"], visual_attention_mask=batch["visual_mask"])

            evidence = batch["evidence"]

            starts = evidence[:, :, 0]
            ends = evidence[:, :, 1]

            start_loss = self.cross_entropy_loss(raw_start, starts[:, 0])  # FIXME: we predict only one evidence.
            end_loss = self.cross_entropy_loss(raw_end, ends[:, 0])

            loss = start_loss + end_loss
        else:
            labels = batch["target_ids"]

            # We clone `label_ids` because `forward` modifies it, but later we need to use it.
            if self.hparams.model_type in {"T5_text_and_visual", "visual_bert_QA"}:
                outputs = self(input_ids=batch["source_ids"], attention_mask=batch["source_mask"],
                               visual=batch["visual_ids"], visual_attention_mask=batch["visual_mask"],
                               labels=labels.clone(), decoder_attention_mask=batch["target_mask"])
            else:
                outputs = self(input_ids=batch["source_ids"], attention_mask=batch["source_mask"],
                               labels=labels.clone(), decoder_attention_mask=batch["target_mask"])

            loss = outputs[0]

            dtype = loss.dtype

            preds = outputs[1].argmax(dim=-1)

            exact_match = (preds == labels).all(dim=-1).to(dtype).mean()  # noqa
            self.log(f"{split}_em", exact_match, prog_bar=True)

        self.log(f"{split}_loss", loss)

        return loss if split == "train" else None

    @overrides(check_signature=False)
    def training_step(self, batch: Mapping[str, Any], batch_idx: int = 0) -> torch.Tensor:
        return self._step(batch, split="train")

    @overrides(check_signature=False)
    def validation_step(self, batch: Mapping[str, Any], batch_idx) -> None:
        return self._step(batch, split="val")

    @overrides(check_signature=False)
    def test_step(self, batch: Mapping[str, Any], batch_idx) -> None:
        return self._step(batch, split="test")

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

        self.trainer.reset_train_dataloader()

        t_total = (
                (len(self.trainer.train_dataloader.loaders)
                 // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=t_total)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    @staticmethod
    def should_log() -> bool:
        return True
