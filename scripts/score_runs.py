"""Aggregate sharded test runs into the challenge score.

Score = mean over episodes of E(D) * success, where
E(D) = 1 - min(max(DTC - DTC_GT, 0) / (NSC_GT - DTC_GT), 1).

test.json ships no reference dialog, so DTC_GT is supplied as a constant that
was calibrated against val_unseen.
"""

import argparse
import json
import os


FIELDS = ("success", "oracle_success", "dtc", "gt_lengths", "trajectory_steps", "le")


def load_runs(dirs, split="test"):
    acc = {name: [] for name in FIELDS}
    for run in dirs:
        path = os.path.join(run, f"metrics_{split}.json")
        metrics = json.load(open(path))["metrics_acc"]
        for name in FIELDS:
            acc[name].extend(metrics[name])
    return acc


def episode_scores(acc, dtc_gt):
    scores = []
    efficiencies = []
    for success, dtc, nsc_gt in zip(acc["success"], acc["dtc"], acc["gt_lengths"]):
        denom = max(nsc_gt - dtc_gt, 1e-9)
        penalty = min(max(dtc - dtc_gt, 0) / denom, 1.0)
        efficiency = 1.0 - penalty
        efficiencies.append(efficiency)
        scores.append(efficiency * float(success))
    return scores, efficiencies


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--dtc_gt", type=float, default=1.0)
    parser.add_argument("--label", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--sweep_dtc_gt",
        action="store_true",
        help="Print the score for a range of DTC_GT values to recover the "
        "constant that reproduces a previously reported score.",
    )
    args = parser.parse_args()

    acc = load_runs(args.runs, args.split)
    count = len(acc["success"])

    if args.sweep_dtc_gt:
        for dtc_gt in (0.0, 1.0, 2.0, 2.5, 3.0):
            scores, effs = episode_scores(acc, dtc_gt)
            print(
                f"dtc_gt={dtc_gt:4.1f}  score={mean(scores):.6f}  "
                f"mean_E(D)={mean(effs):.4f}"
            )
        return

    scores, effs = episode_scores(acc, args.dtc_gt)
    print(
        json.dumps(
            {
                "label": args.label,
                "count": count,
                "sr": 100.0 * mean(acc["success"]),
                "oracle_sr": 100.0 * mean(acc["oracle_success"]),
                "dtc": mean(acc["dtc"]),
                "steps": mean(acc["trajectory_steps"]),
                "le": mean(acc["le"]),
                "mean_efficiency": mean(effs),
                "score": mean(scores),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
