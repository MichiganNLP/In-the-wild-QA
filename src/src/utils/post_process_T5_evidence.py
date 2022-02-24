import argparse
import json
import os

from transformers import AutoTokenizer

from src.evaluations.evaluations import evaluate_evidence
from src.video_qa_with_evidence_dataset import VideoQAWithEvidenceDataModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", help="prediction file path")
    parser.add_argument("--model_name", default="T5_evidence", help="Name of the model, just for output")
    return parser.parse_args()


def post_process(args: argparse.Namespace) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = VideoQAWithEvidenceDataModule(args, tokenizer).test_dataloader().dataset

    with open(args.pred) as file:
        # We only predict one span per instance.
        preds = [[(instance["start"], instance["end"])] for line in file for instance in json.loads(line.strip())]

    evaluate_evidence(args.model_name, preds, dataset)


def main() -> None:
    # FIXME: this needs more arguments to work, probably the same used in __main__.py.
    post_process(parse_args())


if __name__ == "__main__":
    main()
