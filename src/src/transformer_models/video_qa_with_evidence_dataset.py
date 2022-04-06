import argparse
import json
import os, base64, io
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision as TV
import torch.nn.functional as F
import h5py
from cached_path import cached_path
from overrides import overrides
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, CLIPProcessor
from PIL import Image
from collections import defaultdict
from functools import partial

from src.utils.utils import read_hdf5
from src.video_qa_with_evidence_dataset import VideoQAWithEvidenceDataset


def _str2img(b, size_img=224):

    img = Image.open(b).convert('RGB')
    w, h = img.size
    img = TV.transforms.Compose([TV.transforms.Pad([0, (w-h)//2] if w>h else [(h-w)//2, 0]), 
                                    TV.transforms.Resize([size_img, size_img]), 
                                    TV.transforms.ToTensor()])(img)
    return img

def collate(examples: Sequence[Mapping[str, Any]], model_type: str = None, 
            processor: Union[CLIPProcessor, Any] = None) -> Mapping[str, Any]:
    evidence = [ex["evidence"] for ex in examples]
    if evidence[0]:
        max_len = max(len(ev) for ev in evidence)
        evidence_mask = torch.stack([torch.tensor([1] * len(ev) + [0] * (max_len - len(ev))) for ev in evidence])
        evidence = torch.stack([torch.tensor(ev + [(0, 0)] * (max_len - len(ev))) for ev in evidence])
    else:
        evidence = torch.tensor([])
        evidence_mask = torch.tensor([])

    if examples[0]["target_ids"].nelement():
        target_ids = torch.stack([ex["target_ids"] for ex in examples])
        target_mask = torch.stack([ex["target_mask"] for ex in examples])
    else:
        target_ids = torch.tensor([])
        target_mask = torch.tensor([])
    
    if model_type in {"violet_decoder", "clip_decoder", "clip_decoder_eval"}:
        if examples[0]["visual_ids"] and not examples[0]["visual_mask"]:
            visual_ids, visual_masks = [], []
            if model_type == "violet_decoder":
                max_len = max(np.ceil(len(example["visual_ids"]) / 2) for example in examples)
            elif model_type in {"clip_decoder", "clip_decoder_eval"}:
                max_len = max(np.ceil(len(example["visual_ids"]) / 2) for example in examples)
            for example in examples:
                if model_type == "violet_decoder":
                    frame_imgs = [_str2img(img).unsqueeze(0) for img in example["visual_ids"]]
                    catted_imgs = torch.cat(frame_imgs, dim=0)
                    # subsample
                    catted_imgs = catted_imgs[::2, :, :, :]
                    frame_imgs = F.pad(catted_imgs, (0, 0, 0, 0, 0, 0, 0, int(max_len - len(catted_imgs))))
                    
                    visual_ids.append(frame_imgs)
                    visual_masks.append(F.pad(torch.ones(len(catted_imgs)), (0, int(max_len - len(catted_imgs)))))
                elif model_type in {"clip_decoder", "clip_decoder_eval"}:
                    frame_imgs = [Image.open(img) for img in example["visual_ids"]]
                    # not sure here whether we need to cat the images
                    frame_imgs = frame_imgs[::2]
                    processed_imgs = processor(images=frame_imgs, return_tensors="pt")
                    visual_ids.append(processed_imgs['pixel_values'])
                    visual_masks.append(F.pad(torch.ones(len(example["visual_ids"])), (0, int(max_len - len(example["visual_ids"])))))

            return {
                "source_ids": torch.stack([ex["source_ids"] for ex in examples]),
                "source_mask": torch.stack([ex["source_mask"] for ex in examples]),
                "visual_ids": torch.stack([visual_id for visual_id in visual_ids]),
                "visual_mask": torch.stack([visual_mask for visual_mask in visual_masks]),
                "target_ids": target_ids,
                "target_mask": target_mask,
                "evidence": evidence,
                "evidence_mask": evidence_mask,
            }
    
    return {
        "source_ids": torch.stack([ex["source_ids"] for ex in examples]),
        "source_mask": torch.stack([ex["source_mask"] for ex in examples]),
        "visual_ids": torch.stack([ex["visual_ids"] for ex in examples]),
        "visual_mask": torch.stack([ex["visual_mask"] for ex in examples]),
        "target_ids": target_ids,
        "target_mask": target_mask,
        "evidence": evidence,
        "evidence_mask": evidence_mask,
    }


def get_mask_from_sequence_lengths(lengths: torch.Tensor) -> torch.Tensor:
    max_length = int(lengths.max())
    return torch.arange(max_length).expand(len(lengths), max_length) < lengths.unsqueeze(1)  # noqa


class VideoQAWithEvidenceForT5Dataset(VideoQAWithEvidenceDataset):

    def __init__(self, data_dir: str, tokenizer: Union[PreTrainedTokenizerBase, dict], is_test: bool = False,
                 is_zero_shot: bool = False, is_evidence: bool = False, is_multitask: bool = False,
                 include_visual: bool = False, max_len: int = 512, max_vid_len: Optional[int] = None, 
                 path_to_visual_file: Optional[str] = None, visual_size: Optional[int] = None, 
                 path_to_frames: Optional[str] = None, size_img: Optional[int] = None,
                 sample_rate: Optional[int] = None) -> None:
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.is_zero_shot = is_zero_shot
        self.is_evidence = is_evidence
        self.is_multitask = is_multitask

        self.max_len = max_len
        self.include_visual = include_visual
        self.max_vid_len = max_vid_len

        if self.include_visual:
            self.input_visuals = []
            self.evidences = []
            self.frame_paths = {}
            if path_to_frames:
                # read in the frame images in here
                self.size_img = size_img
                self._read_img_paths(path_to_frames)
            else:
                self.visual_features = read_hdf5(cached_path(path_to_visual_file))
                self.sample_rate = sample_rate
                self.visual_size = visual_size
                self._build_visual_features()

                assert all(x.shape[-1] == self.visual_size for x in self.visual_features.values())

        super().__init__(data_dir)

        self.collate_fn = collate

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
    
    def _read_img_paths(self, path):
        domains = os.listdir(path)
        for domain in domains:
            channels = os.listdir(os.path.join(path, domain))
            for channel in channels:
                video_ids = os.listdir(os.path.join(path, domain, channel))
                for video_id in video_ids:
                    imgs = os.listdir(os.path.join(path, domain, channel, video_id))
                    self.frame_paths[video_id] = [os.path.join(path, domain, channel, video_id, img) for img in imgs]

    @overrides
    def _build(self) -> None:
        self._build_examples_from_files()

    def _tokenize(self, text: str, k=None) -> Mapping[str, torch.Tensor]:
        if not k:
            assert isinstance(self.tokenizer, PreTrainedTokenizerBase)
            return self.tokenizer(text, max_length=self.max_len, truncation=True, pad_to_max_length=True,
                              return_tensors="pt")
        else:
            assert isinstance(self.tokenizer, dict)
            return self.tokenizer[k](text, max_length=self.max_len, truncation=True, pad_to_max_length=True,
                              return_tensors="pt")
        

    def _build_examples_from_files(self) -> None:
        with open(self.data_dir) as file:
            data = json.load(file)

        for d in data:
            question = d["question"]
            video_id = d["video_id"]  # For debugging

            k = None
            if self.is_zero_shot:
                question += " <extra_id_0>"
            elif not self.is_evidence:
                if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                    question += f" {self.tokenizer.eos_token}"
                else:
                    assert isinstance(self.tokenizer, dict)
                    question += f" {self.tokenizer['encoder_tokenizer'].eos_token}"
                    k = "encoder_tokenizer"

            tokenized_inputs = self._tokenize(question, k)

            if k:
                k = "decoder_tokenizer"
            if not self.is_test:  # At test time, inputs shouldn't be repeatedly added.
                if self.is_evidence:
                    # Only keep the start and end position as ints.
                    self.evidences.append([[round(float(start_time)), round(float(end_time))]
                                           for evidence in d["evidences"]
                                           for start_time, end_time in evidence.values()])
                elif self.is_multitask:
                    self.evidences.append([[round(float(start_time)), round(float(end_time))]
                                           for evidence in d["evidences"]
                                           for start_time, end_time in evidence.values()])
                    self.targets.append(self._tokenize(f"{d['answer']} {self.tokenizer.eos_token}", k))    
                else:  # not finding evidence, thus tokenize the targets
                    self.targets.append(self._tokenize(f"{d['answer']} {self.tokenizer[k].eos_token}", k))

                # for training, each input and target will be added
            self.inputs.append(tokenized_inputs)

            if self.include_visual:
                if self.frame_paths:
                    self.input_visuals.append({"input_paths": self.frame_paths[video_id]})
                else:
                    self.input_visuals.append({"input_ids": self.visual_features[video_id],
                                           "attention_mask": self.visual_attention_masks[video_id]})

    @overrides
    def __getitem__(self, i: int) -> Mapping[str, Any]:
        source_ids = self.inputs[i]["input_ids"].squeeze()
        src_mask = self.inputs[i]["attention_mask"].squeeze()
        if self.include_visual:
            if self.frame_paths:
                visual_ids = self.input_visuals[i]["input_paths"]
                visual_mask = []
            else:
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
            elif self.is_multitask:
                evidences = self.evidences[i]
                target_ids = self.targets[i]["input_ids"].squeeze()
                target_mask = self.targets[i]["attention_mask"].squeeze()
            else:
                target_ids = self.targets[i]["input_ids"].squeeze()
                target_mask = self.targets[i]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "visual_ids": visual_ids, "visual_mask": visual_mask,
                "target_ids": target_ids, "target_mask": target_mask, "evidence": evidences}


def get_dataset(tokenizer: Union[PreTrainedTokenizerBase, dict], data_dir: str,
                args: argparse.Namespace, is_test: bool = False) -> VideoQAWithEvidenceForT5Dataset:
    if args.model_type in {"T5_text_and_visual", "T5_text_visual_eval"}:
        kwargs = {"include_visual": True, "max_vid_len": args.max_vid_length,
                  "path_to_visual_file": args.path_to_visual_file, "visual_size": args.visual_size,
                  "sample_rate": args.sample_rate}
    elif args.model_type in {"T5_evidence", "T5_evidence_eval", "T5_evidence_IO", "T5_evidence_IO_eval"}:
        kwargs = {"include_visual": True, "max_vid_len": args.max_vid_length,
                  "path_to_visual_file": args.path_to_visual_file, "visual_size": args.visual_size,
                  "sample_rate": args.sample_rate, "is_evidence": True}
    elif args.model_type in {"T5_multi_task", "T5_multi_task_eval"}:
        kwargs = {"include_visual": True, "max_vid_len": args.max_vid_length,
                  "path_to_visual_file": args.path_to_visual_file, "visual_size": args.visual_size,
                  "sample_rate": args.sample_rate, "is_multitask": True}
    elif args.model_type == "violet_decoder":
        # Here the VIOLET will extract visual features by swin transformer,
        # so we only need to provide frame images in this case.
        kwargs = {"include_visual": True, "max_vid_len": args.max_vid_length,
                "path_to_visual_file": None, "sample_rate": args.sample_rate,
                "path_to_frames": args.path_to_frames, "size_img": args.size_img}
    elif args.model_type in {"clip_decoder", "clip_decoder_eval"}:
        kwargs = {"include_visual": True, "max_vid_len": args.max_vid_length,
                "path_to_visual_file": None, "sample_rate": args.sample_rate,
                "path_to_frames": args.path_to_frames}
    elif args.model_type == "T5_zero_shot":
        kwargs = {"is_zero_shot": True}
    else:
        kwargs = {}
    return VideoQAWithEvidenceForT5Dataset(data_dir=data_dir, tokenizer=tokenizer, max_len=args.max_seq_length,
                                           is_test=is_test, **kwargs)


class VideoQAWithEvidenceForT5DataModule(pl.LightningDataModule):  # noqa
    def __init__(self, args: argparse.Namespace, tokenizer: Union[PreTrainedTokenizerBase, dict]) -> None:
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.num_workers = args.num_workers

        self.eval_batch_size = getattr(self.args, "eval_batch_size", getattr(self.args, "batch_size", 1))
        self.model_type = self.args.model_type
        self.processer = None
        if self.model_type in ["clip_decoder", "clip_decoder_eval"]:
            self.processer = CLIPProcessor.from_pretrained(self.args.pretrained_clip_ckpt_path)

    @overrides
    def train_dataloader(self) -> DataLoader:
        dataset = get_dataset(tokenizer=self.tokenizer, data_dir=self.args.train_data, args=self.args)
        return DataLoader(dataset, batch_size=self.args.train_batch_size, drop_last=True, shuffle=True,
                          num_workers=self.num_workers, collate_fn=partial(getattr(dataset, "collate_fn", None), 
                          model_type=self.model_type, processor=self.processer), pin_memory=True, 
                          persistent_workers=self.num_workers > 0)

    @overrides
    def val_dataloader(self) -> DataLoader:
        dataset = get_dataset(tokenizer=self.tokenizer, data_dir=self.args.dev_data, args=self.args)
        return DataLoader(dataset, batch_size=self.eval_batch_size, collate_fn=partial(getattr(dataset, "collate_fn", None), 
                        model_type=self.model_type, processor=self.processer), num_workers=self.num_workers, pin_memory=True, 
                        persistent_workers=self.num_workers > 0)

    @overrides
    def test_dataloader(self) -> DataLoader:
        dataset = get_dataset(tokenizer=self.tokenizer, data_dir=self.args.test_data, args=self.args, is_test=True)
        return DataLoader(dataset, batch_size=self.eval_batch_size, collate_fn=partial(getattr(dataset, "collate_fn", None), 
                        model_type=self.model_type, processor=self.processer), num_workers=self.num_workers, pin_memory=True, 
                        persistent_workers=self.num_workers > 0)
