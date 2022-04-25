from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Callable, NoReturn

import h5py
import pytorch_lightning as pl
import torch
import torchvision.datasets
from cached_path import cached_path
from overrides import overrides
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.datapipes.utils.common import get_file_pathnames_from_root
from torchvision import transforms as T
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import to_tensor
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

from src.transforms import RandomResizedCropWithRandomInterpolation

torchvision.set_image_backend("accimage")


def get_mask_from_sequence_lengths(lengths: torch.Tensor) -> torch.Tensor:
    max_length = int(lengths.max())
    return torch.arange(max_length).expand(len(lengths), max_length) < lengths.unsqueeze(1)  # noqa


def pad_sequence_and_get_mask(sequence: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = pad_sequence(sequence, batch_first=True)  # noqa
    lengths = torch.as_tensor([s.shape[0] for s in sequence])
    return sequence, get_mask_from_sequence_lengths(lengths)


def is_image_file(path: str) -> bool:
    return path.lower().endswith((".bmp", ".jpg", ".jpeg", ".png", ".tiff", ".tif"))


def _get_recursive_image_paths(path: str) -> Iterable[str]:
    for image_path in get_file_pathnames_from_root(path, recursive=True, masks=[]):
        if is_image_file(image_path):
            yield image_path


def _on_walk_error(error: OSError) -> NoReturn:
    warnings.warn(f"{error.filename}: {error.strerror}")
    raise error


def _get_recursive_image_folders(root: str) -> Iterable[str]:
    for path, dirs, filenames in os.walk(root, onerror=_on_walk_error):
        if any(is_image_file(os.path.join(path, filename)) for filename in filenames):
            yield path
        dirs.sort()


class VideoQAWithEvidenceDataset(Dataset):
    def __init__(self, data_path: str, encoder_tokenizer: PreTrainedTokenizerBase | None = None,
                 decoder_tokenizer: PreTrainedTokenizerBase | None = None,
                 transform: Callable[[torch.Tensor], torch.Tensor] | None = None, use_t5_format: bool = False,
                 include_visual: bool = False, max_len: int = 512, max_vid_len: int | None = None,
                 visual_features_path: str | None = None, visual_avg_pool_size: int | None = None,
                 frames_path: str | None = None, fps: int = 3, max_frames: int = 90) -> None:  # TVQA is at 3 fps.
        super().__init__()

        with open(data_path) as file:
            self.instances = json.load(file)

        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_len = max_len  # FIXME: why defining it?
        self.use_t5_format = use_t5_format
        self.transform = transform
        self.max_vid_len = max_vid_len
        self.fps = fps
        self.max_frames = max_frames

        if include_visual:
            if frames_path:
                self.frames_folders_by_video_id = {os.path.basename(video_folder): video_folder
                                                   for video_folder in tqdm(_get_recursive_image_folders(frames_path),
                                                                            desc="Scanning the video frame folders")}

                self.visual_features = None
            elif visual_features_path:
                with h5py.File(cached_path(visual_features_path)) as file:
                    self.visual_features = {v.name.strip("/"): torch.from_numpy(v[:]) for v in file.values()}

                self.frames_folders_by_video_id = None
            else:
                self.visual_features = None
                self.frames_folders_by_video_id = None

            if visual_avg_pool_size:
                self.visual_avg_pool = nn.AvgPool1d(visual_avg_pool_size, ceil_mode=True, count_include_pad=False)
            else:
                self.visual_avg_pool = nn.Identity()
        else:
            self.visual_features = None
            self.frames_folders_by_video_id = None

    @overrides
    def __getitem__(self, i: int) -> Mapping[str, Any]:
        instance = self.instances[i]

        video_id = instance["video_id"]

        output = {
            "id": video_id,
            "question": instance["question"].strip(),
            "answers": [instance["answer"].strip()],
            "evidence": torch.tensor([(round(s), round(e)) for ev in instance["evidences"] for s, e in ev.values()]),
            "duration": float(instance.get("duration", 0.0)),
        }

        if self.visual_features:
            output["visual"] = self.visual_avg_pool(self.visual_features[video_id])[:self.max_vid_len]
        elif self.frames_folders_by_video_id:
            if time_span := instance.get("time_span"):
                start_second, end_second = time_span
                start = math.floor(start_second) * self.fps
                end = (math.ceil(end_second) + 1) * self.fps
            else:
                start = 0
                end = None

            step = 2

            # FIXME: check if `start` > frame files count

            video_folder = self.frames_folders_by_video_id[video_id]

            frame_filenames = sorted(os.listdir(video_folder))[start:end:step][:self.max_frames]
            frames_tensor = torch.stack([to_tensor(default_loader(os.path.join(video_folder, frame_filename)))
                                         for frame_filename in frame_filenames])
            # frames_tensor = frames_tensor.permute(0, 2, 3, 1)
            output["visual"] = self.transform(frames_tensor)

        # TODO: add an option to load directly from the videos.

        return output

    def collate(self, instances: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        keys = next(iter(instances), {})
        batch = {k: [instance[k] for instance in instances] for k in keys}

        for k in keys:
            stack = batch[k]

            if k in {"question", "answers"}:
                if k == "answers":
                    stack = [answers[0] for answers in stack]  # We tokenize only the first answer.
                    k = "answer"

                    tokenizer = self.decoder_tokenizer
                elif k == "question":
                    tokenizer = self.encoder_tokenizer
                else:
                    raise ValueError(f"Unsupported key {k}")

                if tokenizer:
                    if self.use_t5_format and k == "question":
                        # It's more efficient to use the private attribute (`_additional_special_tokens`) than the
                        # public one.
                        to_tokenize = [f"{q} {tokenizer._additional_special_tokens[0]}" for q in stack]
                    elif self.use_t5_format and k == "answers":
                        to_tokenize = [f"<extra_id_0> {a} <extra_id_1>" for a in stack]
                    else:
                        to_tokenize = stack

                    tokenization = tokenizer(to_tokenize, max_length=self.max_len, truncation=True, padding=True,
                                             return_tensors="pt")
                    batch[f"{k}_ids"] = tokenization["input_ids"]
                    batch[f"{k}_mask"] = tokenization["attention_mask"]

        batch["evidence"], batch["evidence_mask"] = pad_sequence_and_get_mask(batch["evidence"])

        if "visual" in keys:
            batch["visual"], batch["visual_mask"] = pad_sequence_and_get_mask(batch["visual"])

        return batch

    def __len__(self) -> int:
        return len(self.instances)


def precision_to_dtype(precision: str | int) -> torch.dtype:
    if precision == 32:
        return torch.float
    elif precision == 64:
        return torch.float64
    elif precision in {16, "mixed"}:
        return torch.float16
    else:
        raise ValueError(f"Unsupported precision value: {precision}")


class VideoQAWithEvidenceDataModule(pl.LightningDataModule):  # noqa
    def __init__(self, args: argparse.Namespace, encoder_tokenizer: PreTrainedTokenizerBase | None = None,
                 decoder_tokenizer: PreTrainedTokenizerBase | None = None) -> None:
        super().__init__()
        self.args = args
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.num_workers = args.num_workers

        self.eval_batch_size = getattr(self.args, "eval_batch_size", getattr(self.args, "batch_size", 1))

    @staticmethod
    def _create_dataset(data_path: str, args: argparse.Namespace,
                        encoder_tokenizer: PreTrainedTokenizerBase | None = None,
                        decoder_tokenizer: PreTrainedTokenizerBase | None = None,
                        transform: Callable[[torch.Tensor], torch.Tensor] | None = None) -> VideoQAWithEvidenceDataset:
        include_visual = args.model_type.startswith(("t5_text_and_visual", "t5_evidence", "t5_multi_task", "clip_"))
        return VideoQAWithEvidenceDataset(data_path=data_path, encoder_tokenizer=encoder_tokenizer,
                                          decoder_tokenizer=decoder_tokenizer, transform=transform,
                                          max_len=getattr(args, "max_seq_length", 512),
                                          max_vid_len=getattr(args, "max_vid_length", None),
                                          visual_avg_pool_size=getattr(args, "visual_avg_pool_size", None),
                                          visual_features_path=getattr(args, "visual_features_path", None),
                                          frames_path=getattr(args, "frames_path", None),
                                          include_visual=include_visual,
                                          use_t5_format=args.model_type == "t5_zero_shot")

    def _create_data_loader(self, data_path: str, is_train: bool = True) -> DataLoader:
        dtype = precision_to_dtype(self.trainer.precision_plugin.precision)

        image_size = getattr(self.args, "size_img", 224)

        if is_train:
            transform = T.Compose([
                # ConvertBHWCtoBCHW(),
                T.ConvertImageDtype(dtype),
                RandomResizedCropWithRandomInterpolation(image_size, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(),
                T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ])
        else:
            transform = T.Compose([
                # ConvertBHWCtoBCHW(),
                T.ConvertImageDtype(dtype),
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ])

        dataset = self._create_dataset(encoder_tokenizer=self.encoder_tokenizer,
                                       decoder_tokenizer=self.decoder_tokenizer, transform=transform,
                                       data_path=data_path, args=self.args)
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
