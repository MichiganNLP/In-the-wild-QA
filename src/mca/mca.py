
from torch.utils.data import ConcatDataset
import numpy as np

from dataloader import VQADataset
from evaluations.evaluations import Evaluation
from tqdm import tqdm
from collections import defaultdict


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
    
    sources = [itm['source'] for itm in test_data]
    labels = [[itm['target']] for itm in test_data]
    evl = Evaluation(sources, preds, labels)    
    
    print("------------------Most Common Ans Text Baseline----------------------")
    print(f"Exact Match: {round(evl.exact_match() * 100, 2)}%")
    print(f"BLEU 1: {round(evl.BLEU(1) * 100, 2)}%")
    print(f"BLEU 2: {round(evl.BLEU(2) * 100, 2)}%")
    print(f"BLEU 3: {round(evl.BLEU(3) * 100, 2)}%")
    print(f"ROUGE 1: {round(evl.ROUGE(1) * 100, 2)}%")
    print(f"ROUGE 1: {round(evl.ROUGE(2) * 100, 2)}%")
    print(f"ROUGE 3: {round(evl.ROUGE(3) * 100, 2)}%")
