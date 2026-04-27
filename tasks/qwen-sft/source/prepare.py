"""One-time data prep for the qwen-sft data-mix search task.

Downloads `allenai/tulu-3-sft-mixture` from Hugging Face and buckets the
per-sample `source` field into five domain axes the manager will mix:

  math      mathematical reasoning + word problems
  code      code generation + completion
  chat      open-ended conversation, persona dialogue
  if        instruction-following with explicit constraints
  reasoning multi-step reasoning, logic, planning

Each bucket is split 95/5 train/val. Held-out val is balanced (200 per
bucket = 1000 samples total) so the manager cannot game the metric by
upweighting the easiest bucket.

Output layout:
  ~/.cache/qwen-sft/data/<bucket>/train.jsonl    Tulu chat format: {"messages": [...]}
  ~/.cache/qwen-sft/data/<bucket>/val.jsonl      held-out 200 samples
  ~/.cache/qwen-sft/data/manifest.json           per-bucket counts + bucketing rules

Run once:
    uv run python prepare.py

Skips already-prepared buckets when their files exist.
"""

import argparse
import json
import os
import random
from pathlib import Path

DATASET_NAME = "allenai/tulu-3-sft-mixture"
SPLIT = "train"
DEFAULT_DATA_ROOT = Path(os.path.expanduser("~/.cache/qwen-sft/data"))

# Tulu-3 source-field substrings → bucket. First match wins.
# Order matters: more specific substrings first.
SOURCE_BUCKETS = {
    "math": [
        "personahub_math",
        "numinamath",
        "tulu_3_persona_math",
        "metamath",
        "math_grade",
    ],
    "code": [
        "tulu_3_persona_code",
        "evol_codealpaca",
        "code_",
        "magicoder",
    ],
    "chat": [
        "wildchat",
        "openassistant",
        "lmsys",
        "no_robots",
        "tulu_hard_coded_repeated_10",
    ],
    "if": [
        "if_",
        "tulu_3_persona_if",
        "ultrachat",
        "personahub_ifdata",
    ],
    "reasoning": [
        "aya",
        "flan",
        "persona_algebra",
        "tulu_3_hardcoded",
        "table_gpt",
        "sciriff",
    ],
}

VAL_PER_BUCKET = 200
TRAIN_CAP_PER_BUCKET = None  # None = keep all training samples


def bucket_for_source(source_str):
    """Return the first bucket whose patterns match this Tulu source string,
    or None if nothing matches (sample is dropped)."""
    if not isinstance(source_str, str):
        return None
    s = source_str.lower()
    for bucket, patterns in SOURCE_BUCKETS.items():
        for pat in patterns:
            if pat in s:
                return bucket
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                        help="Where bucketed jsonl + manifest land. Default: ~/.cache/qwen-sft/data")
    parser.add_argument("--mode", choices=["full", "smoke"], default="full",
                        help="full = whole Tulu mix; smoke = 10K samples for fast plumbing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true",
                        help="Re-bucket even if output files already exist")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # If everything is already on disk and not --force, skip the heavy lifting.
    manifest_path = data_root / "manifest.json"
    expected = [data_root / b / "train.jsonl" for b in SOURCE_BUCKETS]
    expected += [data_root / b / "val.jsonl" for b in SOURCE_BUCKETS]
    if manifest_path.exists() and all(p.exists() for p in expected) and not args.force:
        print(f"prepare.py: data already present at {data_root}; skipping. "
              f"Use --force to re-bucket.")
        return

    # Lazy import — datasets is heavy.
    from datasets import load_dataset

    print(f"prepare.py: streaming {DATASET_NAME} ({SPLIT})…")
    stream = load_dataset(DATASET_NAME, split=SPLIT, streaming=(args.mode == "smoke"))

    # Group by bucket in memory. Tulu-3 mixture is ~5 GB; that fits.
    buckets = {b: [] for b in SOURCE_BUCKETS}
    skipped_unmatched = 0
    skipped_malformed = 0
    cap = 10_000 if args.mode == "smoke" else None

    for i, sample in enumerate(stream):
        if cap is not None and i >= cap:
            break
        if i % 50_000 == 0 and i > 0:
            counts = {b: len(v) for b, v in buckets.items()}
            print(f"  …seen {i:>9,d}; bucket counts so far: {counts}")
        source = sample.get("source")
        bucket = bucket_for_source(source)
        if bucket is None:
            skipped_unmatched += 1
            continue
        msgs = sample.get("messages")
        if not isinstance(msgs, list) or not msgs:
            skipped_malformed += 1
            continue
        buckets[bucket].append({"messages": msgs, "source": source})

    print(f"prepare.py: ingested. unmatched={skipped_unmatched:,d}, "
          f"malformed={skipped_malformed:,d}")
    print(f"prepare.py: per-bucket sizes: "
          f"{ {b: len(v) for b, v in buckets.items()} }")

    manifest = {"dataset": DATASET_NAME, "mode": args.mode,
                "val_per_bucket": VAL_PER_BUCKET, "seed": args.seed,
                "buckets": {}}

    for bucket, samples in buckets.items():
        bucket_dir = data_root / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)

        if len(samples) < VAL_PER_BUCKET + 100:
            print(f"  WARN: bucket {bucket!r} has only {len(samples)} samples; "
                  f"val will be smaller than {VAL_PER_BUCKET}")

        rng.shuffle(samples)
        # Take val from the head, train from the rest.
        n_val = min(VAL_PER_BUCKET, max(1, len(samples) // 20))
        val = samples[:n_val]
        train = samples[n_val:]
        if TRAIN_CAP_PER_BUCKET is not None:
            train = train[:TRAIN_CAP_PER_BUCKET]

        with open(bucket_dir / "train.jsonl", "w", encoding="utf-8") as f:
            for s in train:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        with open(bucket_dir / "val.jsonl", "w", encoding="utf-8") as f:
            for s in val:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        manifest["buckets"][bucket] = {
            "train": len(train),
            "val": len(val),
            "patterns": SOURCE_BUCKETS[bucket],
        }
        print(f"  {bucket:<10s}  train={len(train):>7,d}  val={len(val):>4,d}  "
              f"-> {bucket_dir}")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"prepare.py: done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
