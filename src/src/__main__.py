from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from typing import Callable

from transformers import HfArgumentParser
from transformers.hf_argparser import DataClassType

from src.closest_rtr.closest_rtr import closest_rtr
from src.mca.mca import most_common_ans
from src.parse_args import CLIPDecoderTrainArguments, ClosestRetrievalArguments, DataPathArguments, MODEL_CHOICES, \
    RandomEvidenceArguments, T5EvidenceFindingTrainArguments, T5EvidenceIOTrainArguments, T5MultiTaskTrainArguments, \
    T5TextVisualTrainArguments, T5TrainArguments, T5ZeroShotArguments, VIOLETDecoderTrainArguments, WandbArguments
from src.rdm.random_evidence import random_evidence
from src.rdm.random_text import random_text
from src.transformer_models.eval import transformer_eval
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

    # QA models
    if model_type == "random_text":
        run_model(DataPathArguments, random_text, "random_text")
    elif model_type == "most_common_ans":
        run_model(DataPathArguments, most_common_ans, "most_common_ans")
    elif model_type == "closest_rtr":
        run_model([DataPathArguments, ClosestRetrievalArguments], closest_rtr, "closest_rtr")
    elif model_type == "T5_train":
        run_model([DataPathArguments, T5TrainArguments, WandbArguments], transformer_train, "T5_train")
    elif model_type == "T5_zero_shot":
        run_model([DataPathArguments, T5ZeroShotArguments], transformer_eval, "T5_zero_shot")  # FIXME
    elif model_type == "T5_text_and_visual":
        run_model([DataPathArguments, T5TextVisualTrainArguments, WandbArguments], transformer_train,
                  "T5_text_and_visual")
    # Evidence models
    elif model_type == "random_evidence":
        run_model([DataPathArguments, RandomEvidenceArguments], random_evidence, "random_evidence")
    elif model_type == "T5_evidence":
        run_model([DataPathArguments, T5EvidenceFindingTrainArguments, WandbArguments], transformer_train,
                  "T5_evidence")
    elif model_type == "T5_evidence_IO":
        run_model([DataPathArguments, T5EvidenceIOTrainArguments, WandbArguments], transformer_train, "T5_evidence_IO")
    # Multi-Task models
    elif model_type == "T5_multi_task":
        run_model([DataPathArguments, T5MultiTaskTrainArguments, WandbArguments], transformer_train, "T5_multi_task")
    # Pre-trained models
    elif model_type == "violet_decoder":
        run_model([DataPathArguments, VIOLETDecoderTrainArguments, WandbArguments], transformer_train, "violet_decoder")
    elif model_type == "clip_decoder":
        run_model([DataPathArguments, CLIPDecoderTrainArguments, WandbArguments], transformer_train, "clip_decoder")
    else:
        raise ValueError(f"Unknown model type, you need to pick from {', '.join(MODEL_CHOICES)}")


if __name__ == "__main__":
    main()
