import os
import sys

Root = os.path.dirname(os.path.abspath(__file__))
Dir = os.path.join(Root, "02_core")
if Dir not in sys.path:
    sys.path.append(Dir)

import argparse

from utils.pre_proc import main as preprocess_main
from utils.train import train_main
from utils.inference import inference_main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["preprocess", "train", "infer", "all"],
        help=(
            "preprocess: build features & graph only; "
            "train: train GNN only; "
            "infer: run inference & build submission only; "
            "all: preprocess -> train -> infer"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode in ["preprocess", "all"]:
        preprocess_main()

    if args.mode in ["train", "all"]:
        train_main()

    if args.mode in ["infer", "all"]:
        inference_main()
