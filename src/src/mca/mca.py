from collections import defaultdict

from src.vqa_dataset import VQADataset
from src.evaluations.evaluations import evaluate_qa


def most_common_ans(args):
    train_data = VQADataset(args.train_data)

    # NOTE: here we only use train data as the corpus
    # as dev data is the same as test data in our testing
    # train_dev_data = ConcatDataset([train_data, dev_data])

    test_data = VQADataset(args.test_data)

    ans = defaultdict(int)
    for data in train_data:
        target = data["target"]
        ans[target] += 1

    sorted_ans = sorted(ans.items(), key=lambda kv: kv[1], reverse=True)
    mca = sorted_ans[0][0]

    preds = [mca for _ in range(len(test_data))]

    evaluate_qa("Most Common Ans Text", preds, test_data)
