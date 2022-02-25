import argparse
from collections import defaultdict

from src.evaluations.evaluations import evaluate_qa
from src.video_qa_with_evidence_dataset import VideoQAWithEvidenceDataset


def most_common_ans(args: argparse.Namespace) -> None:
    train_data = VideoQAWithEvidenceDataset(args.train_data)

    # NOTE: here we only use train data as the corpus
    # as dev data is the same as test data in our testing
    # train_dev_data = ConcatDataset([train_data, dev_data])

    test_data = VideoQAWithEvidenceDataset(args.test_data)

    ans = defaultdict(int)
    for data in train_data:
        target = data["target"]
        ans[target] += 1

    sorted_ans = sorted(ans.items(), key=lambda kv: kv[1], reverse=True)
    mca = sorted_ans[0][0]

    preds = [mca] * len(test_data)

    evaluate_qa("Most Common Ans Text", preds, test_data)
