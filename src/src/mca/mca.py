import argparse
from collections import defaultdict

from src.evaluations.evaluations import evaluate_qa
from src.video_qa_with_evidence_dataset import VideoQAWithEvidenceDataset


def most_common_ans(args: argparse.Namespace) -> None:
    train_dataset = VideoQAWithEvidenceDataset(args.train_data_path)

    test_dataset = VideoQAWithEvidenceDataset(args.test_data_path)  # FIXME

    ans = defaultdict(int)
    for data in train_dataset:
        target = data["target"]
        ans[target] += 1

    sorted_ans = sorted(ans.items(), key=lambda kv: kv[1], reverse=True)
    mca = sorted_ans[0][0]

    preds = [mca] * len(test_dataset)

    evaluate_qa("Most Common Ans Text", preds, test_dataset)
