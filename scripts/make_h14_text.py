"""Text embeddings of the guide's fixed vocabulary under ViT-H/14.

The guide needs a second opinion on whether a word is really visible at the
goal, and that opinion is worthless if it comes from the same encoder the
navigator decodes with.  ViT-H/14 (LAION-2B) is an independent witness, but
open_clip cannot share a process with the evaluation stack, so the only thing
computed here is the embedding of a fixed 98 word vocabulary.  That is a model
constant: it carries nothing about any episode, scene, or goal.
"""

import argparse
import ast
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab_source",
        default=(
            "/shared_disk/users/root/FAN_compliant_release_v2/holistic/"
            "guide_caption_search.py"
        ),
    )
    parser.add_argument("--model", default="ViT-H-14")
    parser.add_argument("--pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--template", default="a photo of a {}")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_vocab(path):
    """Pull the word tuples out of the guide module without importing it."""
    tree = ast.parse(Path(path).read_text())
    words = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {
                "ROOM_WORDS",
                "OBJECT_WORDS",
            }:
                words[target.id] = list(ast.literal_eval(node.value))
    missing = {"ROOM_WORDS", "OBJECT_WORDS"} - set(words)
    if missing:
        raise SystemExit(f"could not find {sorted(missing)} in {path}")
    return words["ROOM_WORDS"] + words["OBJECT_WORDS"]


def main():
    args = parse_args()
    vocab = read_vocab(args.vocab_source)
    print(f"vocabulary: {len(vocab)} words", flush=True)

    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    texts = [args.template.format(word) for word in vocab]
    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), 64):
            chunk = texts[start : start + 64]
            encoded = model.encode_text(tokenizer(chunk)).float()
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
            vectors.append(encoded.numpy())

    matrix = np.concatenate(vectors).astype(np.float32)
    np.savez(
        args.output,
        words=np.array(vocab, dtype=object),
        vectors=matrix,
        model=args.model,
        pretrained=args.pretrained,
        template=args.template,
    )
    print(f"wrote {args.output} {matrix.shape}", flush=True)


if __name__ == "__main__":
    main()
