from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import h5py
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.datasets
import torchvision.transforms.functional
from cached_path import cached_path
from overrides import overrides
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

torchvision.set_image_backend("accimage")


def get_mask_from_sequence_lengths(lengths: torch.Tensor) -> torch.Tensor:
    max_length = int(lengths.max())
    return torch.arange(max_length).expand(len(lengths), max_length) < lengths.unsqueeze(1)  # noqa


def pad_sequence_and_get_mask(sequence: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = pad_sequence(sequence, batch_first=True)  # noqa
    lengths = torch.as_tensor([s.shape[0] for s in sequence])
    return sequence, get_mask_from_sequence_lengths(lengths)


class VideoQAWithEvidenceDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizerBase, use_t5_format: bool = False,
                 include_visual: bool = False, max_len: int = 512, max_vid_len: int | None = None,
                 visual_features_path: str | None = None, frames_path: str | None = None,
                 visual_avg_pool_size: int | None = None,
                 transform: Callable[[torch.Tensor], torch.Tensor] | None = None) -> None:
        super().__init__()

        with open(data_path) as file:
            self.instances = json.load(file)

        self.tokenizer = tokenizer
        self.max_len = max_len  # FIXME: why defining it?
        self.use_t5_format = use_t5_format

        if include_visual:
            assert not (visual_features_path and frames_path)

            self.max_vid_len = max_vid_len

            if visual_features_path:
                with h5py.File(cached_path(visual_features_path)) as file:
                    self.visual_features = {v.name.strip("/"): torch.from_numpy(v[:]) for v in file.values()}
            else:
                self.visual_features = None

            if frames_path:
                self.transform = transform
                self.frames_path_by_video_id = {video_id: [os.path.join(frames_path, domain, channel, video_id, img)
                                                           for img in os.listdir(os.path.join(frames_path, domain,
                                                                                              channel,
                                                                                              video_id))]
                                                for domain in os.listdir(frames_path)
                                                for channel in os.listdir(os.path.join(frames_path, domain))
                                                for video_id in os.listdir(os.path.join(frames_path, domain, channel))}
            else:
                self.frames_path_by_video_id = None

            if visual_avg_pool_size:
                self.visual_avg_pool = nn.AvgPool1d(visual_avg_pool_size, ceil_mode=True, count_include_pad=False)
            else:
                self.visual_avg_pool = nn.Identity()
        else:
            self.visual_features = None
            self.frames_path_by_video_id = None

    @overrides
    def __getitem__(self, i: int) -> Mapping[str, Any]:
        instance = self.instances[i]

        output = {
            "id": i,
            "question": instance["question"].strip(),
            "answers": [instance["answer"].strip()],
            "evidence": torch.tensor([(round(s), round(e)) for ev in instance["evidences"] for s, e in ev.values()]),
            "duration": float(instance.get("duration", 0.0)),
        }

        video_id = instance["video_id"]

        if self.visual_features:
            output["visual"] = self.visual_avg_pool(self.visual_features[video_id])[:self.max_vid_len]
        elif self.frames_path_by_video_id:
            frames_tensor = torch.stack([torchvision.transforms.functional.to_tensor(torchvision.datasets.folder.default_loader(video_frame_path))
                                         for video_frame_path in self.frames_path_by_video_id[video_id]])
            output["visual"] = self.transform(frames_tensor)  # TODO: continue

        return output

    def collate(self, instances: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        keys = next(iter(instances), {})
        batch = {k: [instance[k] for instance in instances] for k in keys}

        for k in ["question", "answers"]:
            stack = batch[k]

            if self.tokenizer:
                if k == "question":
                    if self.use_t5_format:
                        # It's more efficient to use the private attribute (`_additional_special_tokens`) than the
                        # public one.
                        to_tokenize = [f"{s} {self.tokenizer._additional_special_tokens[0]}" for s in stack]
                    else:
                        to_tokenize = stack
                elif k == "answers":
                    # FIXME: should add extra IDs for the answer if `self.use_t5_format`.
                    to_tokenize = [e[0] for e in stack]  # We tokenize only the first answer.
                    k = "answer"
                else:
                    raise ValueError(f"Unknown key: {k}")

                tokenization = self.tokenizer(to_tokenize, max_length=self.max_len, truncation=True, padding=True,
                                              return_tensors="pt")
                batch[f"{k}_ids"] = tokenization["input_ids"]
                batch[f"{k}_mask"] = tokenization["attention_mask"]

        batch["evidence"], batch["evidence_mask"] = pad_sequence_and_get_mask(batch["evidence"])

        if "visual" in keys:
            batch["visual"], batch["visual_mask"] = pad_sequence_and_get_mask(batch["visual"])

        return batch

    def __len__(self) -> int:
        return len(self.instances)


class VideoQAWithEvidenceDataModule(pl.LightningDataModule):  # noqa
    def __init__(self, args: argparse.Namespace, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.num_workers = args.num_workers

        self.eval_batch_size = getattr(self.args, "eval_batch_size", getattr(self.args, "batch_size", 1))

    @staticmethod
    def _create_dataset(tokenizer: PreTrainedTokenizerBase, data_path: str,
                        args: argparse.Namespace) -> VideoQAWithEvidenceDataset:
        include_visual = args.model_type.startswith(("T5_text_and_visual", "T5_evidence", "T5_multi_task", "clip_",
                                                     "violet_"))
        return VideoQAWithEvidenceDataset(data_path=data_path, tokenizer=tokenizer, max_len=args.max_seq_length,
                                          max_vid_len=getattr(args, "max_vid_length", None),
                                          visual_avg_pool_size=getattr(args, "visual_avg_pool_size", None),
                                          visual_features_path=getattr(args, "visual_features_path", None),
                                          frames_path=getattr(args, "frames_path", None),
                                          include_visual=include_visual,
                                          use_t5_format=args.model_type == "T5_zero_shot")

    def _create_data_loader(self, data_path: str, is_train: bool = True) -> DataLoader:
        dataset = self._create_dataset(tokenizer=self.tokenizer, data_path=data_path, args=self.args)
        return DataLoader(dataset, batch_size=self.args.train_batch_size if is_train else self.eval_batch_size,
                          drop_last=is_train, shuffle=is_train, collate_fn=getattr(dataset, "collate", None),
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0)

    @overrides
    def train_dataloader(self) -> DataLoader:
        return self._create_data_loader(self.args.train_data_path, is_train=True)

    @overrides
    def val_dataloader(self) -> DataLoader:
        return self._create_data_loader(self.args.dev_data_path, is_train=False)

    @overrides
    def test_dataloader(self) -> DataLoader:
        return self._create_data_loader(self.args.test_data_path, is_train=False)
