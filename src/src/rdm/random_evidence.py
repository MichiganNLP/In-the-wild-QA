import random

from src.vqa_dataset import VQADataset
from src.evaluations.evaluations import evaluate_evidence


def _predict_random(d):
    t_1 = random.uniform(0, d["duration"])
    t_2 = random.uniform(0, d["duration"])
    pred_span = [t_1, t_2]
    pred_span.sort()
    return pred_span


def random_evidence(args):
    test_data = VQADataset(args.test_data)

    preds = []
    # evaluate the test data
    for d in test_data:
        prediction = [_predict_random(d) for _ in range(args.pred_num)]
        preds.append(prediction)

    evaluate_evidence("Random Evidence", preds, test_data)
