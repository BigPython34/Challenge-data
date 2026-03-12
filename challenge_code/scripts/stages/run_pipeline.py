#!/usr/bin/env python3
"""Simple CLI to orchestrate the reorganized pipeline stages."""

import argparse
from typing import Sequence

from .optimize_models import main as optimize_main
from .prepare_data import main as prepare_main
from .predict import predict_and_submit
from .train_models import main as train_main
from .train_models_clean import run_clean_training

ALL_STAGES = ["prepare", "train", "train-clean", "optimize", "predict"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one or more stage scripts.")
    parser.add_argument(
        "--stages",
        "-s",
        nargs="+",
        choices=ALL_STAGES + ["all"],
        default=["prepare", "train"],
        help="Stages to execute in order. Use 'all' to run every stage in sequence.",
    )
    parser.add_argument(
        "--clean-outliers",
        type=int,
        default=100,
        help="When running the clean training stage, how many samples to drop.",
    )
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip the interactive prediction stage even if it is requested.",
    )
    return parser.parse_args()


def _expand_stages(requested: Sequence[str]) -> list[str]:
    if "all" in requested:
        return ALL_STAGES.copy()
    result: list[str] = []
    for stage in ALL_STAGES:
        if stage in requested:
            result.append(stage)
    return result


def main() -> None:
    args = parse_args()
    stages = _expand_stages(args.stages)

    for stage in stages:
        if stage == "prepare":
            prepare_main()
        elif stage == "train":
            train_main()
        elif stage == "train-clean":
            run_clean_training(args.clean_outliers)
        elif stage == "optimize":
            optimize_main()
        elif stage == "predict":
            if args.skip_predict:
                print("[INFO] Prediction stage skipped.")
                continue
            predict_and_submit()
        else:
            print(f"[WARN] Unknown stage requested: {stage}")


if __name__ == "__main__":
    main()
