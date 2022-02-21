import argparse
import json
from typing import Any, Mapping

import numpy as np
import torch
from overrides import overrides
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

from src.utils.utils import read_hdf5
from src.vqa_dataset import VQADataset


def get_mask_from_sequence_lengths(lengths: torch.Tensor) -> torch.Tensor:
    max_length = int(lengths.max())
    return torch.arange(max_length).expand(len(lengths), max_length) < lengths.unsqueeze(1)  # noqa


class T5Dataset(VQADataset):
    def __init__(self, data_dir, tokenizer, is_test=False, is_zero_shot=False, is_evidence=False,
                 include_visual=False, max_len=512, max_vid_len=None, path_to_visual_file=None,
                 visual_size=None, sample_rate=None) -> None:
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

            assert all(x.shape[-1] == self.visual_size for _, x in self.visual_features.items())
            self.input_visuals = []
            self.evidences = []

        super().__init__(data_dir)

    def _build_visual_features(self) -> None:
        sample_rate = self.sample_rate

        vfs = self.visual_features.values()

        if sample_rate:
            # Sample video features
            vfs = []
            for x in self.visual_features.values():
                repeat_times = (sample_rate - x.shape[0] % sample_rate) % sample_rate
                supplement_itms = np.repeat(x[-1, :].reshape((1, -1)), repeat_times, axis=0)
                full_x = np.concatenate((x, supplement_itms), axis=0)
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
        return self.tokenizer(text, max_length=self.max_len, pad_to_max_length=True, return_tensors="pt")

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

        return {"source_ids": source_ids,
                "source_mask": src_mask,
                "visual_ids": visual_ids,
                "visual_mask": visual_mask,
                "target_ids": target_ids,
                "target_mask": target_mask,
                "evidence": evidences}


def get_dataset(tokenizer: PreTrainedTokenizerBase, data_dir: str, args: argparse.Namespace) -> T5Dataset:
    if args.model_type == "T5_text_and_visual":
        kwargs = {"max_vid_len": args.max_vid_length, "path_to_visual_file": args.path_to_visual_file,
                  "visual_size": args.visual_size, "sample_rate": args.sample_rate}
    elif args.model_type == "T5_evidence":
        kwargs = {"max_vid_len": args.max_vid_length, "path_to_visual_file": args.path_to_visual_file,
                  "visual_size": args.visual_size, "sample_rate": args.sample_rate, "is_evidence": True}
    else:
        kwargs = {}
    return T5Dataset(data_dir=data_dir, tokenizer=tokenizer, max_len=args.max_seq_length, **kwargs)
