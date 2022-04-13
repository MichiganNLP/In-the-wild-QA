from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from typing import Callable

from transformers import HfArgumentParser
from transformers.hf_argparser import DataClassType

from src.closest_rtr.closest_rtr import closest_rtr
from src.mca.mca import evaluate_most_common_answer
from src.parse_args import CLIPDecoderTrainArguments, ClosestRetrievalArguments, DataPathArguments, MODEL_CHOICES, \
    RandomEvidenceArguments, T5EvidenceFindingTrainArguments, T5EvidenceIOTrainArguments, T5MultiTaskTrainArguments, \
    T5TextVisualTrainArguments, T5TrainArguments, T5ZeroShotArguments, VIOLETDecoderTrainArguments, WandbArguments
from src.rdm.random_evidence import random_evidence
from src.rdm.random_text import random_text
from src.transformer_models.train import transformer_train


def run_model(dataclass_types: DataClassType | Iterable[DataClassType],
              model_function: Callable[[argparse.Namespace], None], model_type: str) -> None:
    # Don't pass a generator here as it misbehaves. See https://github.com/huggingface/transformers/pull/15758
    parser = HfArgumentParser(dataclass_types)

    args_in_dataclasses_and_extra_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    args_in_dataclasses, extra_args = args_in_dataclasses_and_extra_args[:-1], args_in_dataclasses_and_extra_args[-1]

    extra_args.remove(model_type)
    assert not extra_args, f"Unknown arguments: {extra_args}"

    args = argparse.Namespace(**{k: v for args in args_in_dataclasses for k, v in args.__dict__.items()})
    args.model_type = model_type

    model_function(args)


def main() -> None:
    model_type = sys.argv[1]

    if model_type == "random_text":
        dataclass_types = DataPathArguments
        model_function = random_text
    elif model_type == "most_common_ans":
        dataclass_types = DataPathArguments
        model_function = evaluate_most_common_answer
    elif model_type == "closest_rtr":
        dataclass_types = [DataPathArguments, ClosestRetrievalArguments]
        model_function = closest_rtr
    elif model_type == "random_evidence":
        dataclass_types = [DataPathArguments, RandomEvidenceArguments]
        model_function = random_evidence
    elif model_type in MODEL_CHOICES:
        dataclass_types = {
            "T5_train": [DataPathArguments, T5TrainArguments, WandbArguments],
            "T5_zero_shot": [DataPathArguments, T5ZeroShotArguments],
            "T5_text_and_visual": [DataPathArguments, T5TextVisualTrainArguments, WandbArguments],
            "T5_evidence": [DataPathArguments, T5EvidenceFindingTrainArguments, WandbArguments],
            "T5_evidence_IO": [DataPathArguments, T5EvidenceIOTrainArguments, WandbArguments],
            "T5_multi_task": [DataPathArguments, T5MultiTaskTrainArguments, WandbArguments],
            "violet_decoder": [DataPathArguments, VIOLETDecoderTrainArguments, WandbArguments],
            "clip_decoder": [DataPathArguments, CLIPDecoderTrainArguments, WandbArguments],
        }[model_type]
        model_function = transformer_train
    else:
        raise ValueError(f"Unknown model type, you need to pick from: {', '.join(MODEL_CHOICES)}")

    run_model(dataclass_types, model_function, model_type)


if __name__ == "__main__":
    main()
