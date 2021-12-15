import argparse
import json

from src.dataloader import VQADataset
from src.evaluations.evaluations import evidence_evaluation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", help="prediction file path")
    parser.add_argument("--processed_pred", help="processed file (output file) path")
    parser.add_argument("--test_data", help="path to the test data")
    parser.add_argument("--model_name", default="T5_evidence",
        type=str, help="Name of the model, just for output")
    args = parser.parse_args()

    return args


def post_process(args):
    with open(args.pred, 'r') as f:
        data = f.readlines()
    data = [json.loads(d.strip()) for d in data]

    processed_data = []
    # here we only predict one span for an instance
    for instance in data:
        processed_data.append([[instance["start"], instance["end"]]])
    
    test_data = VQADataset(args.test_data)

    evidence_evaluation(f"{args.model_name}", processed_data, test_data)


if __name__ == "__main__":
    args = parse_args()
    post_process(args)
