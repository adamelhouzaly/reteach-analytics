"""Command-line entry points.

    python -m reteach.cli generate-data
    python -m reteach.cli train
    python -m reteach.cli evaluate
"""

from __future__ import annotations

import argparse
import json

from . import RANDOM_SEED


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="reteach")
    parser.add_argument("command", choices=["generate-data", "train", "evaluate"])
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)

    if args.command == "generate-data":
        from .datagen import generate_all
        print(json.dumps(generate_all(args.seed), indent=2))
    elif args.command == "train":
        from .evaluate import train_model
        trained, info = train_model(args.seed)
        print(f"Saved model (C={info['C']}, review threshold={trained.threshold:.2f}) "
              f"trained on {trained.meta['n_train']} dev examples.")
    else:
        from .evaluate import run_evaluation
        metrics = run_evaluation(args.seed)
        t = metrics["test"]
        print(f"Test accuracy {t['model']['accuracy']:.3f}, macro F1 {t['model']['macro_f1']:.3f} "
              f"(baseline accuracy {t['baseline_most_frequent']['accuracy']:.3f}, "
              f"macro F1 {t['baseline_most_frequent']['macro_f1']:.3f}).")
        s = t["selective_prediction"]
        print(f"Threshold {s['threshold']:.2f}: coverage {s['coverage']:.1%}, "
              f"accepted accuracy {s['accepted_accuracy']}")
        print("Wrote artifacts/metrics.json and artifacts/evaluation_report.md")


if __name__ == "__main__":
    main()
