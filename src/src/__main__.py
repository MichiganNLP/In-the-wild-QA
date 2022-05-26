from __future__ import annotations

import argparse
import sys

from transformers import HfArgumentParser

from src.parse_args import model_type_to_dataclass_types
from src.train_and_test import train_and_test


def main() -> None:
    model_type = sys.argv[1]
    dataclass_types = model_type_to_dataclass_types(model_type)

    # Don't pass a generator here as it misbehaves. See https://github.com/huggingface/transformers/pull/15758
    parser = HfArgumentParser(dataclass_types)

    args_in_dataclasses_and_extra_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    args_in_dataclasses, extra_args = args_in_dataclasses_and_extra_args[:-1], args_in_dataclasses_and_extra_args[-1]

    extra_args.remove(model_type)
    assert not extra_args, f"Unknown arguments: {extra_args}"

    args = argparse.Namespace(**{k: v for args in args_in_dataclasses for k, v in args.__dict__.items()})
    args.model_type = model_type

    train_and_test(args)


if __name__ == "__main__":
    main()
