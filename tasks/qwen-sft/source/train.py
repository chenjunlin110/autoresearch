"""Full-FT SFT trainer for Qwen3-0.6B on a weighted mix of Tulu-3 buckets.

The manager LLM proposes new mixture weights by editing the module-level
``DATA_MIX`` dict (via the framework's `param_patch` + `constant_replace`
edit kind). Everything else here is a fixed search baseline:

  - Qwen3-0.6B in bf16, full fine-tuning (no LoRA / no PEFT).
  - Manual training loop with a weighted multi-source sampler, AdamW,
    grad-ckpt + flash_attention_2 if available (falls back to SDPA).
  - Train until either ``SFT_TIME_BUDGET_SECONDS`` (default 600 = 10 min)
    of wall time has elapsed, or 5,000 steps have run, whichever comes first.
  - Final ``val_loss`` is the *balanced* mean cross-entropy across all
    five buckets' held-out samples — gaming the metric by upweighting an
    easy bucket doesn't help.

Outputs ``metrics.json`` with at minimum ``val_loss`` (finite float).
The framework's result-validator reads this; the manager optimizes it.

Run:
    SFT_TIME_BUDGET_SECONDS=600 \\
    QWEN_SFT_METRICS_PATH=/abs/path/metrics.json \\
    uv run python train.py
"""

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Manager-editable knobs (the data-mix is the only one that should change).
# ---------------------------------------------------------------------------

# Manager's data-mix decision lives here. Keys must match bucket names from
# prepare.py's manifest.json. Values are sampling weights; they will be
# normalized to sum to 1.
DATA_MIX = {
    "math":      0.20,
    "code":      0.20,
    "chat":      0.20,
    "if":        0.20,
    "reasoning": 0.20,
}

DATA_ROOT      = os.path.expanduser("~/.cache/qwen-sft/data")
# Qwen3-0.6B for demo: ~1.2 GB bf16, full FT comfortably under 12 GB on
# H200. ~5-10× faster wall per experiment than 4B — we get more search
# iterations per hour, which is what the data-mix search actually needs.
# Bigger micro_batch (16 vs 4) since memory pressure is gone, and we drop
# grad_accum=1 since one micro covers the effective batch we want.
MODEL_NAME     = "Qwen/Qwen3-0.6B"
SEQ_LEN        = 2048
MICRO_BATCH    = 16
GRAD_ACCUM     = 1
LEARNING_RATE  = 1e-5
WARMUP_STEPS   = 50
WEIGHT_DECAY   = 0.0
GRAD_CKPT      = True
MAX_STEPS      = 5000
LOG_EVERY      = 10

# 10 min default for demo — 0.6B trains fast enough that 10 min wastes
# budget on an already-converged model. Override with SFT_TIME_BUDGET_SECONDS.
TIME_BUDGET = int(os.environ.get("SFT_TIME_BUDGET_SECONDS", "600"))
DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Metrics output
# ---------------------------------------------------------------------------

def write_metrics(metrics):
    metrics_path = os.environ.get("QWEN_SFT_METRICS_PATH", "").strip()
    if not metrics_path:
        return
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_bucket_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def render_with_assistant_loss_mask(messages, tokenizer, seq_len):
    """Tokenize a chat-formatted Tulu sample with assistant-only loss masking.

    Returns ``(input_ids, labels)`` of length ``seq_len`` (padded with the
    pad token; labels for non-assistant tokens are set to -100 so loss is
    not computed on them, matching standard SFT).
    """
    # Apply chat template once for prompt-only ids (everything before the
    # last assistant turn) and once for the full conversation. The diff is
    # the assistant target.
    if not messages or messages[-1].get("role") != "assistant":
        return None
    prompt_messages = messages[:-1]
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
    )
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True,
    )
    if len(full_ids) > seq_len:
        # Truncate from the front so we keep the assistant target.
        excess = len(full_ids) - seq_len
        full_ids = full_ids[excess:]
        prompt_ids = prompt_ids[excess:] if len(prompt_ids) > excess else []

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = list(full_ids)
    labels = list(full_ids)
    # Mask everything up to and including the prompt with -100.
    mask_until = min(len(prompt_ids), len(labels))
    for i in range(mask_until):
        labels[i] = -100

    pad = seq_len - len(input_ids)
    if pad > 0:
        input_ids += [pad_id] * pad
        labels += [-100] * pad

    return input_ids, labels


class WeightedBucketSampler:
    """Loads each bucket's train.jsonl into memory, pre-tokenizes each
    sample on first access, and yields batches by sampling a bucket
    proportionally to ``DATA_MIX``."""

    def __init__(self, mix, data_root, tokenizer, seq_len, split="train"):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.buckets = list(mix.keys())
        weights = np.array([mix[b] for b in self.buckets], dtype=np.float64)
        if weights.sum() <= 0:
            raise ValueError("DATA_MIX weights must sum to a positive number.")
        self.weights = weights / weights.sum()
        self.samples = {b: load_bucket_jsonl(Path(data_root) / b / f"{split}.jsonl")
                         for b in self.buckets}
        # Pre-tokenize on demand; avoids heavy upfront cost when buckets are large.
        self._cache = {b: [None] * len(self.samples[b]) for b in self.buckets}
        for b, lst in self.samples.items():
            if not lst:
                raise FileNotFoundError(
                    f"bucket {b!r} {split} split is empty at {data_root}/{b}/{split}.jsonl. "
                    f"Run prepare.py first.")
        self.rng = np.random.default_rng(seed=42)

    def _tokenize_one(self, bucket, idx):
        cached = self._cache[bucket][idx]
        if cached is not None:
            return cached
        sample = self.samples[bucket][idx]
        ids = render_with_assistant_loss_mask(sample["messages"], self.tokenizer, self.seq_len)
        if ids is None:
            ids = (None, None)  # mark dead
        self._cache[bucket][idx] = ids
        return ids

    def next_batch(self, batch_size):
        xs, ys = [], []
        attempts = 0
        while len(xs) < batch_size:
            attempts += 1
            if attempts > batch_size * 50:
                raise RuntimeError(
                    "Sampler couldn't assemble a batch — too many degenerate samples?")
            bucket = self.rng.choice(self.buckets, p=self.weights)
            idx = self.rng.integers(0, len(self.samples[bucket]))
            input_ids, labels = self._tokenize_one(bucket, idx)
            if input_ids is None:
                continue
            xs.append(input_ids)
            ys.append(labels)
        x = torch.tensor(xs, dtype=torch.long, device=DEVICE)
        y = torch.tensor(ys, dtype=torch.long, device=DEVICE)
        return x, y


def balanced_val_batch(buckets, data_root, tokenizer, seq_len, max_per_bucket=200):
    """Build the held-out balanced val tensors once. Returns
    ``(input_ids[N, seq_len], labels[N, seq_len])`` with samples drawn
    evenly from each bucket so the metric isn't gameable by upweighting an
    easy bucket."""
    xs, ys = [], []
    for bucket in buckets:
        bucket_path = Path(data_root) / bucket / "val.jsonl"
        if not bucket_path.exists():
            continue
        samples = load_bucket_jsonl(bucket_path)[:max_per_bucket]
        for s in samples:
            ids = render_with_assistant_loss_mask(s["messages"], tokenizer, seq_len)
            if ids is None:
                continue
            xs.append(ids[0])
            ys.append(ids[1])
    if not xs:
        raise RuntimeError(f"No valid val samples found under {data_root}.")
    return (torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading tokenizer + model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try flash-attn first; fall back to SDPA if not installed (some cluster
    # nodes can't compile flash-attn). transformers handles this transparently
    # — SDPA is ~10-20% slower but doesn't change the data-mix metric.
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except Exception:
        attn_impl = "sdpa"
    print(f"loading model with attn_implementation={attn_impl}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    ).to(DEVICE)
    if GRAD_CKPT:
        model.gradient_checkpointing_enable()
    model.train()

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: total={num_params/1e9:.2f}B trainable={num_trainable/1e9:.2f}B")
    print(f"DATA_MIX (normalized): "
          f"{ {k: round(v / sum(DATA_MIX.values()), 4) for k, v in DATA_MIX.items()} }")

    print("building train sampler…")
    sampler = WeightedBucketSampler(DATA_MIX, DATA_ROOT, tokenizer, SEQ_LEN, split="train")
    print("building val tensors…")
    val_x, val_y = balanced_val_batch(list(DATA_MIX.keys()), DATA_ROOT, tokenizer, SEQ_LEN)
    print(f"val: {val_x.shape[0]} samples")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
        fused=True,
    )

    def lr_at(step):
        if step < WARMUP_STEPS:
            return LEARNING_RATE * (step + 1) / WARMUP_STEPS
        return LEARNING_RATE

    t_start = time.time()
    last_log = t_start
    step = 0
    micros = 0
    train_loss_running = 0.0
    train_loss_count = 0
    optimizer.zero_grad(set_to_none=True)

    while step < MAX_STEPS:
        elapsed = time.time() - t_start
        if elapsed >= TIME_BUDGET:
            print(f"step {step}: hit time budget {TIME_BUDGET}s — stopping")
            break

        x, y = sampler.next_batch(MICRO_BATCH)
        out = model(input_ids=x, labels=y)
        loss = out.loss / GRAD_ACCUM
        loss.backward()
        train_loss_running += out.loss.item()
        train_loss_count += 1
        micros += 1

        if micros % GRAD_ACCUM == 0:
            for pg in optimizer.param_groups:
                pg["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % LOG_EVERY == 0:
                avg = train_loss_running / max(1, train_loss_count)
                now = time.time()
                dt = now - last_log
                last_log = now
                print(f"step {step:05d} ({elapsed:.0f}s/{TIME_BUDGET}s) | "
                      f"loss: {avg:.4f} | lr: {lr_at(step):.2e} | "
                      f"dt: {dt*1000/LOG_EVERY:.0f}ms",
                      flush=True)
                train_loss_running = 0.0
                train_loss_count = 0

    train_seconds = time.time() - t_start
    print(f"training done after {step} steps / {train_seconds:.0f}s")

    # ----------------- final balanced val loss -----------------
    print("evaluating balanced val…")
    model.eval()
    val_loss_sum = 0.0
    val_token_count = 0
    eval_bs = MICRO_BATCH
    with torch.no_grad():
        for i in range(0, val_x.shape[0], eval_bs):
            xb = val_x[i:i+eval_bs].to(DEVICE)
            yb = val_y[i:i+eval_bs].to(DEVICE)
            logits = model(input_ids=xb).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = yb[:, 1:].contiguous()
            mask = shift_labels != -100
            if mask.sum() == 0:
                continue
            losses = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            )
            val_loss_sum += losses.sum().item()
            val_token_count += int(mask.sum().item())
    val_loss = val_loss_sum / max(1, val_token_count)
    print(f"val_loss: {val_loss:.4f}  (over {val_token_count} target tokens)")

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    write_metrics({
        "val_loss":             float(val_loss),
        "training_seconds":     float(train_seconds),
        "total_seconds":        float(time.time() - t_start),
        "num_steps":            int(step),
        "num_params":           int(num_params),
        "num_params_trainable": int(num_trainable),
        "peak_vram_mb":         float(peak_vram_mb),
        "data_mix":             dict(DATA_MIX),
        "data_mix_normalized":  {k: v / sum(DATA_MIX.values()) for k, v in DATA_MIX.items()},
        "model_name":           MODEL_NAME,
        "seq_len":              int(SEQ_LEN),
        "micro_batch":          int(MICRO_BATCH),
        "grad_accum":           int(GRAD_ACCUM),
        "learning_rate":        float(LEARNING_RATE),
        "time_budget_seconds":  int(TIME_BUDGET),
        "device_name":          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    })


if __name__ == "__main__":
    main()
