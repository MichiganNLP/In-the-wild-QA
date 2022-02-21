import argparse
import random
from typing import Any, Iterable, Mapping

from src.evaluations.evaluations import evaluate_evidence
from src.vqa_dataset import VQADataset


def _predict_random(d: Mapping[str, Any]) -> Iterable[float]:
    t_1 = random.uniform(0, d["duration"])
    t_2 = random.uniform(0, d["duration"])
    pred_span = [t_1, t_2]
    return sorted(pred_span)


def random_evidence(args: argparse.Namespace) -> None:
    test_data = VQADataset(args.test_data)
    preds = [[_predict_random(d) for _ in range(args.pred_num)] for d in test_data]
    evaluate_evidence("Random Evidence", preds, test_data)
