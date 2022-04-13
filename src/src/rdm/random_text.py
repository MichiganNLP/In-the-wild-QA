import argparse
import random

from src.evaluations.evaluations import evaluate_qa
from src.video_qa_with_evidence_dataset import VideoQAWithEvidenceDataModule


def random_text(args: argparse.Namespace) -> None:
    data_module = VideoQAWithEvidenceDataModule(args)
    train_dataset = data_module.train_dataloader()
    test_dataset = data_module.test_dataloader()

    preds = random.choices([target_instance
                            for batch in train_dataset
                            for target_instance in batch["target"]], k=len(test_dataset))  # noqa

    evaluate_qa("Random Text", preds, test_dataset)
