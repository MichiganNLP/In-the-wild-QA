import argparse
import random

from torch.utils.data import ConcatDataset

from src.evaluations.evaluations import evaluate_qa
from src.video_qa_with_evidence_dataset import VideoQAWithEvidenceDataset


def random_text(args: argparse.Namespace) -> None:
    # FIXME
    train_data_path = VideoQAWithEvidenceDataset(args.train_data_path)
    dev_data_path = VideoQAWithEvidenceDataset(args.dev_data_path)

    train_dev_dataset = ConcatDataset([train_data_path, dev_data_path])

    train_dataset = VideoQAWithEvidenceDataset(args.test_data_path)

    # evaluate the test data
    preds = random.choices(train_dev_dataset, k=len(train_dataset))  # noqa
    preds = [pred["target"] for pred in preds]

    evaluate_qa("Random Text", preds, train_dataset)
