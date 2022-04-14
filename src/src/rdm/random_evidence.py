import argparse
import os
import random
from collections.abc import Iterable, Mapping
from typing import Any

from transformers import AutoTokenizer

from src.evaluations.evaluations import evaluate_evidence
from src.video_qa_with_evidence_dataset import VideoQAWithEvidenceDataModule


def _predict_random(d: Mapping[str, Any]) -> Iterable[float]:
    t_1 = random.uniform(0, d["duration"])
    t_2 = random.uniform(0, d["duration"])
    pred_span = [t_1, t_2]
    return sorted(pred_span)


def evaluate_random_evidence(args: argparse.Namespace) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = VideoQAWithEvidenceDataModule(args, tokenizer).test_dataloader().dataset

    preds = [[_predict_random(d) for _ in range(args.pred_num)] for d in dataset]
    evaluate_evidence("Random Evidence", preds, dataset)
