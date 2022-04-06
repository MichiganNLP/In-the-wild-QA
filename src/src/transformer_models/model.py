from typing import Any, Literal, Mapping, Optional, Union, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from overrides import overrides
from torch.optim import AdamW
from transformers import PreTrainedTokenizerBase, T5Config, T5ForConditionalGeneration, \
    VisualBertForQuestionAnswering, get_linear_schedule_with_warmup
from transformers.models.t5.modeling_t5 import T5EncoderModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from collections import defaultdict

from src.transformer_models.t5_and_visual import T5AndVisual, TextVisualEncoder, _combine_attention_masks
from src.transformer_models.violet_decoder.model import VioletWithDecoder
from src.transformer_models.clip_decoder import CLIPWithDecoder
from src.utils.utils import device


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
        self.linear = nn.Linear(self.config.d_model, 1)

    @overrides(check_signature=False)
    def forward(self, masked_caption_ids: Optional[torch.Tensor] = None, visual: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None, visual_attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(masked_caption_ids, visual=visual, attention_mask=attention_mask,
                               visual_attention_mask=visual_attention_mask, **kwargs)
        visual_start = masked_caption_ids.shape[1]
        visual_hidden = outputs.last_hidden_state[:, visual_start:, :]
        transformed_hidden = self.linear(visual_hidden) # reduce feature dimension to 1
        prob_in = torch.sigmoid(transformed_hidden[..., 0])   # calculate the probability that the frame is "I"
        return prob_in

    def predict(self, masked_caption_ids: Optional[torch.Tensor] = None, visual: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                visual_attention_mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        prob_in = self(masked_caption_ids, visual, attention_mask, visual_attention_mask)
        in_index = (prob_in >= 0.5).nonzero(as_tuple=False).cpu()   # move variables to cpu to append to list
        results = defaultdict(list)
        # get the position for each frame that is within the evidence
        for [batch_idx, frame_idx] in in_index:
            batch_idx, frame_idx = int(batch_idx), int(frame_idx)
            if not results[batch_idx]:
                results[batch_idx].append([frame_idx, frame_idx])
            else:
                last_frame_idx = results[batch_idx][-1][1]
                if last_frame_idx + 1 == frame_idx:
                    results[batch_idx][-1][1] = frame_idx
                else:
                    results[batch_idx].append([frame_idx, frame_idx])

        results_with_score = defaultdict(list)
        # get the score of the evidence
        for b, start_ends in results.items():
            for [start_idx, end_idx] in start_ends:
                score = torch.mean(prob_in[b][start_idx: end_idx + 1]).cpu()
                results_with_score[b].append([start_idx, end_idx, float(score)])
        
        # sort the evidence based on the score
        for b in results_with_score:
            results_with_score[b].sort(key=lambda x: x[-1], reverse=True)
        return results_with_score

class T5MultiTask(T5ForConditionalGeneration):

    def __init__(self, config: T5Config, visual_size: int) -> None:
        super().__init__(config)
        self.encoder = TextVisualEncoder(self.encoder, visual_size)  # noqa
        self.start_end = nn.Linear(self.config.d_model, 2)

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
        start = end = None
        if "encoder_outputs" not in kwargs:
            # only here when doing the multi-task training
            start, end, encoder_outputs = self._evidence_forward(masked_caption_ids, visual=visual, attention_mask=attention_mask,
                                visual_attention_mask=visual_attention_mask, **kwargs)
            kwargs["encoder_outputs"] = encoder_outputs

        # The attention mask used in the encoder (the combined mask) is actually necessary for the decoder even when
        # providing "encoder_outputs". We could compute it once in the encoder, return it as one of its fields and use
        # it here. However, in `T5FillerModel.generative_step` we input the encoder outputs but without the mask
        # since it's constructed from the `generate` output which in turn only returns certain fields (not the mask).
        attention_mask = _combine_attention_masks(attention_mask, visual_attention_mask)
        outputs = super().forward(attention_mask=attention_mask, labels=labels, **kwargs) # noqa

        # add variables on the fly
        if start and end:
            setattr(outputs, "start", start)
            setattr(outputs, "end", end)
        return outputs
    
    def _evidence_forward(self, masked_caption_ids: Optional[torch.Tensor] = None, visual: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None, visual_attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, **kwargs) -> tuple[torch.Tensor, torch.Tensor, Seq2SeqLMOutput]: 
        encoder_outputs = self.encoder(masked_caption_ids, visual=visual, attention_mask=attention_mask,
                                        visual_attention_mask=visual_attention_mask, **kwargs)
        visual_start = masked_caption_ids.shape[1]
        visual_hidden = encoder_outputs.last_hidden_state[:, visual_start:, :]

        start_end = self.start_end(visual_hidden)
        return start_end[..., 0], start_end[..., 1], encoder_outputs

    
    def predict(self, masked_caption_ids: Optional[torch.Tensor] = None, visual: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None, visual_attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        start, end, _ = self._evidence_forward(masked_caption_ids, visual=visual, attention_mask=attention_mask,
                         visual_attention_mask=visual_attention_mask, **kwargs)
        return start.log_softmax(dim=-1), end.log_softmax(dim=-1)


class FineTuner(pl.LightningModule):  # noqa
    def __init__(self, tokenizer: Union[PreTrainedTokenizerBase, dict], **kwargs) -> None:  # noqa
        super().__init__()

        self.tokenizer = tokenizer

        self.save_hyperparameters()

        if self.hparams.model_type == "T5_text_and_visual":
            self.model = T5AndVisual.from_pretrained(self.hparams.model_name_or_path,
                                                     visual_size=self.hparams.visual_size)
        elif self.hparams.model_type == "T5_evidence":
            self.model = T5AndVisualEvidence.from_pretrained(self.hparams.model_name_or_path,
                                                             visual_size=self.hparams.visual_size)
            self.cross_entropy_loss = nn.CrossEntropyLoss()

        elif self.hparams.model_type == "T5_evidence_IO":
            self.model = T5AndVisualEvidenceIO.from_pretrained(self.hparams.model_name_or_path,
                                                                visual_size=self.hparams.visual_size)   
            self.cross_entropy_loss = nn.CrossEntropyLoss()
        elif self.hparams.model_type == "T5_multi_task":
            self.model = T5MultiTask.from_pretrained(self.hparams.model_name_or_path,
                                                    visual_size=self.hparams.visual_size)
            self.cross_entropy_loss = nn.CrossEntropyLoss()
        elif self.hparams.model_type == "violet_decoder":
            self.model = VioletWithDecoder.from_pretrained(self.hparams.model_name_or_path, 
                                        pretrained_violet_ckpt_path=self.hparams.pretrained_violet_ckpt_path)
        elif self.hparams.model_type == "clip_decoder":
            self.model = CLIPWithDecoder.from_pretrained(self.hparams.model_name_or_path,
                                        pretrained_clip_ckpt_path=self.hparams.pretrained_clip_ckpt_path,
                                        max_seq=self.hparams.max_seq_length)
        else:
            assert self.hparams.model_type in ["T5_train", "T5_zero_shot"]
            self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)

    @overrides(check_signature=False)
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                visual: Optional[torch.Tensor] = None, visual_attention_mask: Optional[torch.Tensor] = None,
                decoder_input_ids: Optional[torch.Tensor] = None,
                decoder_attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> Any:
        if labels is not None:
            if isinstance(self.tokenizer, dict):
                labels[labels == self.tokenizer["decoder_tokenizer"].pad_token_id] = -100  # For the loss computation.
            else:
                labels[labels == self.tokenizer.pad_token_id] == -100
        
        if self.hparams.model_type in {"T5_text_and_visual", "T5_multi_task", "violet_decoder", "clip_decoder"}:
            kwargs = {"attention_mask": attention_mask, "visual": visual,
                        "visual_attention_mask": visual_attention_mask, "labels": labels}
        elif self.hparams.model_type in {"T5_evidence", "T5_evidence_IO"}:
            kwargs = {"attention_mask": attention_mask, "visual": visual,
                        "visual_attention_mask": visual_attention_mask}
        else:
            kwargs = {"attention_mask": attention_mask, "labels": labels}

        return self.model(input_ids, **kwargs)
    
    def _evidence_loss(self, raw_start, raw_end, batch):
        evidence = batch["evidence"]

        starts = evidence[:, :, 0]
        ends = evidence[:, :, 1]

        start_loss = self.cross_entropy_loss(raw_start, starts[:, 0])  # FIXME: we predict only one evidence.
        end_loss = self.cross_entropy_loss(raw_end, ends[:, 0])
        
        return start_loss + end_loss

    def _evidence_IO_loss(self, prob_in, batch):
        # construct the groundtruth sequence
        gold = torch.zeros(prob_in.shape)
        evidence = batch["evidence"]
        starts = evidence[:, :, 0]
        ends = evidence[:, :, 1]
        batch_size, evidence_number = starts.shape
        
        for b_idx in range(batch_size):
            for e_idx in range(evidence_number):
                gold[b_idx][starts[b_idx][e_idx]: ends[b_idx][e_idx]] = 1.0
        gold = gold.to(device)
        loss = self.cross_entropy_loss(prob_in, gold)

    def _seq2seq_loss_exact_match(self, batch, outputs, split):
        labels = batch["target_ids"]
        loss = outputs.loss
        dtype = loss.dtype
        preds = outputs[1].argmax(dim=-1)
        exact_match = (preds == labels).all(dim=-1).to(dtype).mean()  # noqa
        self.log(f"{split}_em", exact_match, prog_bar=True)
        return loss

    def _step(self, batch: Mapping[str, Any], split: Literal["train", "val", "test"]) -> Optional[torch.Tensor]:
        # We clone `label_ids` because `forward` modifies it, but later we need to use it.
        outputs = self(input_ids=batch["source_ids"], attention_mask=batch["source_mask"],
                        visual=batch["visual_ids"], visual_attention_mask=batch["visual_mask"],
                        labels=batch["target_ids"].clone(), decoder_attention_mask=batch["target_mask"])
    
        if self.hparams.model_type == "T5_evidence":
            raw_start, raw_end = outputs
            loss = self._evidence_loss(raw_start, raw_end, batch)

            
        elif self.hparams.model_type == "T5_evidence_IO":

            loss = self._evidence_IO_loss(outputs, batch)

        elif self.hparams.model_type == "T5_multi_task":
            evidence_loss = self._evidence_loss(outputs.start, outputs.end, batch)
            seq2seq_loss = self._seq2seq_loss_exact_match(batch, outputs, split)
            loss = self.hparams.vqa_weight * seq2seq_loss + self.hparams.evidence_weight * evidence_loss
        elif self.hparams.model_type in ["violet_decoder", "clip_decoder"]:
            seq2seq_loss = self._seq2seq_loss_exact_match(batch, outputs, split)
            loss = seq2seq_loss
        else:
            loss = self._seq2seq_loss_exact_match(batch, outputs, split)

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
