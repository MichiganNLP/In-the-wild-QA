import argparse
import json
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from overrides import overrides
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from src.utils.utils import read_hdf5
from src.video_qa_with_evidence_dataset import VideoQAWithEvidenceDataset


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


def get_mask_from_sequence_lengths(lengths: torch.Tensor) -> torch.Tensor:
    max_length = int(lengths.max())
    return torch.arange(max_length).expand(len(lengths), max_length) < lengths.unsqueeze(1)  # noqa


class VideoQAWithEvidenceForT5Dataset(VideoQAWithEvidenceDataset):
    def __init__(self, data_dir: str, tokenizer: PreTrainedTokenizerBase, is_test: bool = False,
                 is_zero_shot: bool = False, is_evidence: bool = False, include_visual: bool = False,
                 max_len: int = 512, max_vid_len: Optional[int] = None, path_to_visual_file: Optional[str] = None,
                 visual_size: Optional[int] = None, sample_rate: Optional[int] = None) -> None:
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.is_zero_shot = is_zero_shot
        self.is_evidence = is_evidence

        self.max_len = max_len
        self.include_visual = include_visual
        self.max_vid_len = max_vid_len

        if self.include_visual:
            self.visual_features = read_hdf5(path_to_visual_file)
            self.sample_rate = sample_rate
            self.visual_size = visual_size
            self._build_visual_features()

            assert all(x.shape[-1] == self.visual_size for x in self.visual_features.values())
            self.input_visuals = []
            self.evidences = []

        super().__init__(data_dir)

        self.collate_fn = my_collate

    def _build_visual_features(self) -> None:
        sample_rate = self.sample_rate

        vfs = self.visual_features.values()

        if sample_rate:
            # Sample video features
            vfs = []
            for x in self.visual_features.values():
                repeat_times = (sample_rate - x.shape[0] % sample_rate) % sample_rate
                supplement_items = np.repeat(x[-1, :].reshape((1, -1)), repeat_times, axis=0)
                full_x = np.concatenate((x, supplement_items), axis=0)
                downsampled_x = np.mean(full_x.reshape(-1, sample_rate, self.visual_size), axis=1)  # noqa
                vfs.append(downsampled_x)

        # Trim video features
        processed_features = pad_sequence([torch.tensor(x[:self.max_vid_len, :]) for x in vfs], batch_first=True)
        lengths = torch.tensor([min(itm.shape[0], self.max_vid_len) for itm in vfs])

        attention_masks = get_mask_from_sequence_lengths(lengths)
        assert attention_masks.shape[0] == processed_features.shape[0]
        self.visual_attention_masks = {}
        for i, k in zip(range(processed_features.shape[0]), self.visual_features):
            self.visual_features[k] = processed_features[i]
            self.visual_attention_masks[k] = attention_masks[i]

    @overrides
    def _build(self) -> None:
        self._build_examples_from_files()

    def _tokenize(self, text: str) -> Mapping[str, torch.Tensor]:
        return self.tokenizer(text, max_length=self.max_len, truncation=True, pad_to_max_length=True,
                              return_tensors="pt")

    def _build_examples_from_files(self) -> None:
        with open(self.data_dir) as file:
            data = json.load(file)

        for d in data:
            question = d["question"]
            video_id = d["video_id"]  # For debugging

            if self.is_zero_shot:
                question += " <extra_id_0>"
            elif not self.is_evidence:
                question += f" {self.tokenizer.eos_token}"

            tokenized_inputs = self._tokenize(question)

            if not self.is_test:  # At test time, inputs shouldn't be repeatedly added.
                if self.is_evidence:
                    # Only keep the start and end position as ints.
                    self.evidences.append([[round(float(start_time)), round(float(end_time))]
                                           for evidence in d["evidences"]
                                           for start_time, end_time in evidence.values()])
                else:  # not finding evidence, thus tokenize the targets
                    self.targets.append(self._tokenize(f"{d['answer']} {self.tokenizer.eos_token}"))

                # for training, each input and target will be added
            self.inputs.append(tokenized_inputs)

            if self.include_visual:
                self.input_visuals.append({"input_ids": self.visual_features[video_id],
                                           "attention_mask": self.visual_attention_masks[video_id]})

    @overrides
    def __getitem__(self, i: int) -> Mapping[str, Any]:
        source_ids = self.inputs[i]["input_ids"].squeeze()
        src_mask = self.inputs[i]["attention_mask"].squeeze()

        if self.include_visual:
            visual_ids = self.input_visuals[i]["input_ids"].squeeze()
            visual_mask = self.input_visuals[i]["attention_mask"].squeeze()
        else:
            visual_ids = torch.tensor([])
            visual_mask = torch.tensor([])

        target_ids = torch.tensor([])
        target_mask = torch.tensor([])
        evidences = []

        if not self.is_test:
            if self.is_evidence:
                evidences = self.evidences[i]
            else:
                target_ids = self.targets[i]["input_ids"].squeeze()
                target_mask = self.targets[i]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "visual_ids": visual_ids, "visual_mask": visual_mask,
                "target_ids": target_ids, "target_mask": target_mask, "evidence": evidences}


def get_dataset(tokenizer: PreTrainedTokenizerBase, data_dir: str,
                args: argparse.Namespace, is_test: bool = False) -> VideoQAWithEvidenceForT5Dataset:
    if args.model_type == "T5_text_and_visual":
        kwargs = {"include_visual": True, "max_len": args.max_seq_length, "max_vid_len": args.max_vid_length,
                  "path_to_visual_file": args.path_to_visual_file, "visual_size": args.visual_size, "sample_rate":
                      args.sample_rate}
    elif args.model_type == "T5_evidence":
        kwargs = {"include_visual": True, "max_len": args.max_seq_length, "max_vid_len": args.max_vid_length,
                  "path_to_visual_file": args.path_to_visual_file, "visual_size": args.visual_size, "sample_rate":
                      args.sample_rate, "is_evidence": True}
    elif args.model_type == "T5_zero_shot":
        kwargs = {"is_zero_shot": True}
    else:
        kwargs = {}
    return VideoQAWithEvidenceForT5Dataset(data_dir=data_dir, tokenizer=tokenizer, max_len=args.max_seq_length,
                                           is_test=is_test, **kwargs)


class VideoQAWithEvidenceForT5DataModule(pl.LightningDataModule):  # noqa
    def __init__(self, args: argparse.Namespace, tokenizer: PreTrainedTokenizerBase, num_workers: int = 4) -> None:
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.num_workers = num_workers

    @overrides
    def train_dataloader(self) -> DataLoader:
        dataset = get_dataset(tokenizer=self.tokenizer, data_dir=self.args.train_data, args=self.args)
        return DataLoader(dataset, batch_size=self.args.train_batch_size, drop_last=True, shuffle=True,
                          num_workers=self.num_workers, collate_fn=getattr(dataset, "collate_fn", None),
                          pin_memory=True, persistent_workers=self.num_workers > 0)

    @overrides
    def val_dataloader(self) -> DataLoader:
        dataset = get_dataset(tokenizer=self.tokenizer, data_dir=self.args.dev_data, args=self.args)
        return DataLoader(dataset, batch_size=getattr(self.args, "eval_batch_size", self.args.batch_size),
                          collate_fn=getattr(dataset, "collate_fn", None), num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=self.num_workers > 0)

    @overrides
    def test_dataloader(self) -> DataLoader:
        dataset = get_dataset(tokenizer=self.tokenizer, data_dir=self.args.test_data, args=self.args, is_test=True)
        return DataLoader(dataset, batch_size=getattr(self.args, "eval_batch_size", self.args.batch_size),
                          collate_fn=getattr(dataset, "collate_fn", None), num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=self.num_workers > 0)
