import argparse
from collections import Counter

from src.evaluations.evaluations import evaluate_qa
from src.video_qa_with_evidence_dataset import VideoQAWithEvidenceDataModule


def evaluate_most_common_answer(args: argparse.Namespace) -> None:
    data_module = VideoQAWithEvidenceDataModule(args)
    train_dataset = data_module.train_dataloader()
    test_dataset = data_module.test_dataloader()

    answer_counts = Counter(target_instance
                            for batch in train_dataset
                            for target_instance in batch["answer"])

    preds = [answer_counts.most_common(n=1)[0][0]] * len(test_dataset)

    evaluate_qa("Most Common Ans Text", preds, test_dataset)
