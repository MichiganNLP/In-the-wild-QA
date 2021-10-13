import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

from T5.T5_dataloader import T5Dataset, get_dataset


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()

        self.save_hyperparameters(hparams)

        self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.tokenizer_name_or_path)
    
    def is_logger(self):
        return self.trainer.global_rank <= 0
    
    def forward(
        self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
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