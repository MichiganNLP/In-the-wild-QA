import random
import sys

from argparse import Namespace
from transformers import HfArgumentParser

from src.parse_args import (
    MODEL_CHOICES,
    DataPathArguments, 
    ClostRtrModelArguments, 
    T5TrainArguments,
    T5EvalArguments,
    T5ZeroShotArguments,
    T5TextVisualTrainArguments,
    T5TextVisualEvalArguments,
    RandomEvidenceArguments,
    T5EvidenceFindingTrainArguments,
    T5EvidenceFindingEvalArguments,
    WandbArguments
)

from src.rdm.random_text import random_text
from src.rdm.random_evidence import random_evidence
from src.closest_rtr.closest_rtr import closest_rtr
from src.mca.mca import most_common_ans
from src.transformer_models.train import transformer_train
from src.transformer_models.eval import transformer_eval



def run_model(Arguments: list, model_function, model_type: str):
    parser = HfArgumentParser((Argument for Argument in Arguments))
    args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    args.model_type = model_type
    model_function(args)


def main():
    model_type = sys.argv[1]
    if model_type not in MODEL_CHOICES:
        raise ValueError(f"Unkown model type, you need to pick from {', '.join(MODEL_CHOICES)}")

    # QA models
    if model_type == "random_text":
        run_model([DataPathArguments], random_text, "random_text")

    elif model_type == "most_common_ans":
        run_model([DataPathArguments], most_common_ans, "most_common_ans")

    elif model_type == "closest_rtr":
        run_model([DataPathArguments, ClostRtrModelArguments], closest_rtr, "closest_rtr")

    elif model_type == "T5_train":
        run_model([DataPathArguments, T5TrainArguments, WandbArguments], transformer_train, "T5_train")
       
    elif model_type == "T5_eval":
        run_model([DataPathArguments, T5EvalArguments], transformer_eval, "T5_eval")

    elif model_type == "T5_zero_shot":
        run_model([DataPathArguments, T5ZeroShotArguments], transformer_eval, "T5_zero_shot")

    elif model_type == "T5_text_and_visual":
        run_model([DataPathArguments, T5TextVisualTrainArguments, WandbArguments], transformer_train, "T5_text_and_visual")

    elif model_type == "T5_text_visual_eval":
        run_model([DataPathArguments, T5TextVisualEvalArguments], transformer_eval, "T5_text_visual_eval")

    # Evidence models
    elif model_type == "random_evidence":
        run_model([DataPathArguments, RandomEvidenceArguments], random_evidence, "random_evidence")

    elif model_type == "T5_evidence":
        run_model([DataPathArguments, T5EvidenceFindingTrainArguments, WandbArguments], transformer_train, "T5_evidence")

    elif model_type == "T5_evidence_eval":
        run_model([DataPathArguments, T5EvidenceFindingEvalArguments], transformer_eval, "T5_evidence_eval")


if __name__ == "__main__":
    main()