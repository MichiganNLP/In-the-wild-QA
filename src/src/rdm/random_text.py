import random
from torch.utils.data import ConcatDataset

from src.dataloader import VQADataset
from src.evaluations.evaluations import evaluate


def random_text(args):
    train_data = VQADataset(args.train_data)
    dev_data = VQADataset(args.dev_data)

    train_dev_data = ConcatDataset([train_data, dev_data])

    test_data = VQADataset(args.test_data)

    # evaluate the test data
    preds = random.choices(train_dev_data, k=len(test_data))
    preds = [pred['target'] for pred in preds]

    evaluate("Random Text", preds, test_data)
