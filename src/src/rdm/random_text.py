import argparse
import random

from torch.utils.data import ConcatDataset

from src.evaluations.evaluations import evaluate_qa
from src.video_qa_with_evidence_dataset import VideoQAWithEvidenceDataset


def random_text(args: argparse.Namespace) -> None:
    train_data = VideoQAWithEvidenceDataset(args.train_data)
    dev_data = VideoQAWithEvidenceDataset(args.dev_data)

    train_dev_data = ConcatDataset([train_data, dev_data])

    test_data = VideoQAWithEvidenceDataset(args.test_data)

    # evaluate the test data
    preds = random.choices(train_dev_data, k=len(test_data))  # noqa
    preds = [pred["target"] for pred in preds]

    evaluate_qa("Random Text", preds, test_data)
