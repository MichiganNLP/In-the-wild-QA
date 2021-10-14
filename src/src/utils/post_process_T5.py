import argparse
from dataloader import VQADataset
from evaluations.evaluations import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", help="prediction file path")
    parser.add_argument("--processed_pred", help="processed file (output file) path")
    parser.add_argument("--test_data", help="path to the test data")
    args = parser.parse_args()

    return args


def post_process(args):
    with open(args.pred, 'r') as f:
        data = f.readlines()

    processed_data = []
    for d in data:
        d = d.split("</s>")[0]
        d = d.split("<pad> ")[-1]
        processed_data.append(d)
    
    with open(args.processed_pred, 'w') as f:
        f.write("\n".join(processed_data))
    
    test_data = VQADataset(args.test_data)

    evaluate("T5 Text", processed_data, test_data)


if __name__ == "__main__":
    args = parse_args()
    post_process(args)
