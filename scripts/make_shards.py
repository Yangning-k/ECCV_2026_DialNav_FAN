"""Split an official annotation file into the eight shards the runner expects.

Evaluation is parallelised across eight GPUs, one shard each.  The split is
contiguous and deterministic, so the union is exactly the official file and the
per-episode results do not depend on how many shards are used.
"""

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation",
        required=True,
        help="e.g. $DATA_ROOT/dataset/RAIN_holistic/val_unseen.json",
    )
    parser.add_argument("--split", required=True,
                        choices=["val_seen", "val_unseen", "test"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shards", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.annotation) as handle:
        episodes = json.load(handle)
    if not isinstance(episodes, list):
        episodes = list(episodes.values())

    os.makedirs(args.output_dir, exist_ok=True)
    total = len(episodes)
    for index in range(args.shards):
        start = int(index * total / args.shards)
        end = int((index + 1) * total / args.shards)
        path = os.path.join(
            args.output_dir, f"{args.split}{args.shards}_{index}.json"
        )
        with open(path, "w") as handle:
            json.dump(episodes[start:end], handle)
        print(f"{path}: {end - start} episodes", flush=True)
    print(f"{total} episodes over {args.shards} shards", flush=True)


if __name__ == "__main__":
    main()
