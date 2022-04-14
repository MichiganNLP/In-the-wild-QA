from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from typing import Callable

from transformers import HfArgumentParser
from transformers.hf_argparser import DataClassType

from src.closest_rtr.closest_rtr import evaluate_closest_rtr
from src.mca.mca import evaluate_most_common_answer
from src.parse_args import MODEL_CHOICES, \
    model_type_to_dataclass_types
from src.rdm.random_evidence import evaluate_random_evidence
from src.rdm.random_text import evaluate_random_text
from src.transformer_models.train import train_transformer


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


def model_type_to_function(model_type: str) -> Callable[[argparse.Namespace], None]:
    if model_type == "random_text":
        return evaluate_random_text
    elif model_type == "most_common_ans":
        return evaluate_most_common_answer
    elif model_type == "closest_rtr":
        return evaluate_closest_rtr
    elif model_type == "random_evidence":
        return evaluate_random_evidence
    elif model_type in MODEL_CHOICES:
        return train_transformer
    else:
        raise ValueError(f"Unknown model type, you need to pick from: {', '.join(MODEL_CHOICES)}")


def main() -> None:
    model_type = sys.argv[1]
    dataclass_types = model_type_to_dataclass_types(model_type)
    model_function = model_type_to_function(model_type)
    run_model(dataclass_types, model_function, model_type)


if __name__ == "__main__":
    main()
