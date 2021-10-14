import random
from torch.utils.data import ConcatDataset

from dataloader import VQADataset
from evaluations.evaluations import Evaluation


def random_text(args):
    train_data = VQADataset(args.train_data)
    dev_data = VQADataset(args.dev_data)

    train_dev_data = ConcatDataset([train_data, dev_data])

    test_data = VQADataset(args.test_data)

    # evaluate the test data
    preds = random.choices(train_dev_data, k=len(test_data))
    preds = [pred['target'] for pred in preds]

    sources = [itm['source'] for itm in test_data]
    labels = [[itm['target']] for itm in test_data]
    evl = Evaluation(sources, preds, labels)    
    
    print("------------------Random Text Baseline----------------------")
    print(f"Exact Match: {round(evl.exact_match() * 100, 2)}%")
    print(f"BLEU 1: {round(evl.BLEU(1) * 100, 2)}%")
    print(f"BLEU 2: {round(evl.BLEU(2) * 100, 2)}%")
    print(f"BLEU 3: {round(evl.BLEU(3) * 100, 2)}%")
    print(f"ROUGE 1: {round(evl.ROUGE(1) * 100, 2)}%")
    print(f"ROUGE 1: {round(evl.ROUGE(2) * 100, 2)}%")
    print(f"ROUGE 3: {round(evl.ROUGE(3) * 100, 2)}%")
