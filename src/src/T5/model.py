import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from overrides import overrides
import torch.nn as nn

from typing import Any, Mapping, Optional, Tuple, Union

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from transformers import T5Config, T5ForConditionalGeneration

from src.T5.T5_dataloader import T5Dataset, get_dataset



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


class TextVisualEncoder(T5Stack):
    def __init__(self, t5stack: T5Stack, visual_size: int) -> None:
        super().__init__(t5stack.config, t5stack.embed_tokens)
        self.embed_video = nn.Linear(visual_size, self.embed_tokens.embedding_dim)

    @overrides
    def forward(self, text_token_ids: torch.Tensor, visual: torch.Tensor,  # noqa
                attention_mask: Optional[torch.Tensor] = None, visual_attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Union[BaseModelOutputWithPastAndCrossAttentions, Tuple[torch.Tensor, ...]]:
        text_embedding = self.embed_tokens(text_token_ids)
        visual_embedding = self.embed_video(visual)
        embedding = torch.cat([text_embedding, visual_embedding], dim=1)
        attention_mask = _combine_attention_masks(attention_mask, visual_attention_mask)

        return super().forward(inputs_embeds=embedding, attention_mask=attention_mask, **kwargs)


class T5AndVisual(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, visual_size: int) -> None:
        super().__init__(config)
        self.encoder = TextVisualEncoder(self.encoder, visual_size)

    @overrides
    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, visual: Optional[torch.Tensor] = None,
                                      visual_attention_mask: Optional[torch.Tensor] = None,
                                      **kwargs) -> Mapping[str, Any]:
        output = super().prepare_inputs_for_generation(input_ids, **kwargs)  # noqa
        output["visual"] = visual
        output["visual_attention_mask"] = visual_attention_mask
        return output

    @overrides
    def forward(self, masked_caption_ids: Optional[torch.Tensor] = None, visual: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None, visual_attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, **kwargs) -> Union[Seq2SeqLMOutput, Tuple[torch.Tensor, ...]]:
        if "encoder_outputs" not in kwargs:
            kwargs["encoder_outputs"] = self.encoder(masked_caption_ids, visual=visual, attention_mask=attention_mask,
                                                     visual_attention_mask=visual_attention_mask, **kwargs)

        # The attention mask used in the encoder (the combined mask) is actually necessary for the decoder even when
        # providing "encoder_outputs". We could compute it once in the encoder, return it as one of its fields and use
        # it here. However, in `T5FillerModel.generative_step` we input the encoder outputs but without the mask
        # since its constructed from the `generate` output which in turn only returns certain fields (not the mask).
        attention_mask = _combine_attention_masks(attention_mask, visual_attention_mask)

        return super().forward(attention_mask=attention_mask, labels=labels, **kwargs)  # noqa



class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()

        self.save_hyperparameters(hparams)

        if self.hparams.model_type != "T5_text_and_visual":
            self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
        else:
            self.model = T5AndVisual.from_pretrained(self.hparams.model_name_or_path, visual_size=self.hparams.visual_size)

        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.tokenizer_name_or_path)
    
    def is_logger(self):
        return self.trainer.global_rank <= 0
    
    def forward(
        self, input_ids, attention_mask=None, visual=None, visual_attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
    ):  
        if self.hparams.model_type != "T5_text_and_visual":
            return self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
            )
        else:
            return self.model(
                input_ids,
                attention_mask=attention_mask,
                visual=visual,
                visual_attention_mask=visual_attention_mask,
                labels=labels,
            )

    def _step(self, batch):
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        if self.hparams.model_type != "T5_text_and_visual":
            outputs = self(
                    input_ids=batch["source_ids"],
                    attention_mask=batch["source_mask"],
                    labels=labels,
                    decoder_attention_mask=batch['target_mask']
            )
        else:
            outputs = self(
                    input_ids=batch["source_ids"],
                    attention_mask=batch["source_mask"],
                    visual=batch["visual_ids"],
                    visual_attention_mask=batch["visual_mask"],
                    labels=labels,
                    decoder_attention_mask=batch['target_mask']
            )

        loss = outputs[0]

        preds = torch.argmax(outputs[1].softmax(-1), dim=-1)
        acc = torch.eq(preds, labels).sum() / (labels != -100).sum()  # token level acc
        N, l = preds.shape 
        sent_acc = []
        for i in np.arange(N):
            p = preds[i, :]
            t = labels[i, :]
            p[t[:] == -100] = -100
            sent_acc.append(torch.equal(p, t))
        sent_acc = sum(sent_acc) / len(sent_acc)
        return loss, acc, sent_acc

    def training_step(self, batch, batch_idx):
        loss, acc, sent_acc = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True)
        self.log('train_sentence_acc', sent_acc, on_step=True, on_epoch=True)
        return {"loss": loss, "log": tensorboard_logs}
    
    def training_epoch_end(self, outputs):
        # if outputs:
        #TODO: fix this hacky solution
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        # self.log('epoch_train_loss', avg_train_loss)
        # self.log('epoch_train_acc', self.accuracy.compute(), on_step=False, on_epoch=True)
        # return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, acc, sent_acc = self._step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True)
        self.log('val_sentence_acc', sent_acc, on_step=True, on_epoch=True)
        return {"val_loss": loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        # self.log('avg_val_loss', avg_loss)
        # self.log('avg_val_acc', self.accuracy.compute())
        # return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, on_tpu=None, using_native_amp=None, using_lbfgs=None):
        # if self.hparams.use_tpu:
        #     xm.optimizer_step(optimizer)
        # else:
        # NOTE: not using tpu
        assert not self.hparams.use_tpu
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()
    
    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, data_dir=self.hparams.train_data, args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, data_dir=self.hparams.dev_data, args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)
    