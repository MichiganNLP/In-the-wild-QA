import argparse

from transformers import AutoTokenizer

from src.evaluations.evaluations import evaluate_qa
from src.video_qa_with_evidence_dataset import VideoQAWithEvidenceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", help="prediction file path")
    parser.add_argument("--processed_pred", help="processed file (output file) path")
    parser.add_argument("--dataset", help="path to the test data")
    parser.add_argument("--model_name_or_path", default="t5-base")
    parser.add_argument("--model_type", default="T5_Text", help="Model type, just for output")
    return parser.parse_args()


def post_process(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    extra_id_0_token = tokenizer.additional_special_tokens[0]

    with open(args.pred) as file:
        processed_data = [generated_answer.split(tokenizer.eos_token,
                                                 maxsplit=1)[0].rsplit(tokenizer.pad_token,
                                                                       maxsplit=1)[-1].rsplit(extra_id_0_token,
                                                                                              maxsplit=1)[-1].strip()
                          for generated_answer in file]

    with open(args.processed_pred, "w") as file:
        file.write("\n".join(processed_data))

    dataset = VideoQAWithEvidenceDataset(args.test_data_path)  # FIXME

    evaluate_qa(args.model_type, processed_data, dataset)


def main() -> None:
    post_process(parse_args())


if __name__ == "__main__":
    main()
