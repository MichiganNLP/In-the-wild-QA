import argparse
import random

from parse_args import parse_args
from rdm.random_text import random_text
from closest_rtr.closest_rtr import closest_rtr
from mca.mca import most_common_ans
from T5.train import T5_train
from T5.eval import T5_eval


def main():
    args = parse_args()
    random.seed(args.random_state)
    if args.model_type == 'random_text':
        random_text(args)
    elif args.model_type == "most_common_ans":
        most_common_ans(args)
    elif args.model_type == "closest_rtr":
        closest_rtr(args)
    elif args.model_type == "T5_train":
        T5_train(args)
    elif args.model_type == "T5_eval":
        T5_eval(args)
    elif args.model_type == "T5_zero_shot":
        T5_eval(args)


if __name__ == "__main__":
    main()