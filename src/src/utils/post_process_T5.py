import argparse

from src.evaluations.evaluations import evaluate_qa
from src.vqa_dataset import VQADataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", help="prediction file path")
    parser.add_argument("--processed_pred", help="processed file (output file) path")
    parser.add_argument("--test_data", help="path to the test data")
    parser.add_argument("--model_name", default="T5_Text",
                        type=str, help="Name of the model, just for output")
    args = parser.parse_args()

    return args


def post_process(args):
    with open(args.pred) as f:
        data = f.readlines()

    processed_data = []
    for d in data:
        d = d.split("</s>")[0]
        d = d.split("<pad> ")[-1]
        d = d.split("<extra_id_0>")[-1]
        processed_data.append(d)

    with open(args.processed_pred, "w") as f:
        f.write("\n".join(processed_data))

    test_data = VQADataset(args.test_data)

    evaluate_qa(f"{args.model_name}", processed_data, test_data)


if __name__ == "__main__":
    args = parse_args()
    post_process(args)
