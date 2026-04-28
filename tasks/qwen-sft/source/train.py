"""Full-FT SFT trainer for Qwen3-0.6B on a weighted mix of Tulu-3 buckets.

The manager LLM proposes new mixture weights by editing the module-level
``DATA_MIX`` dict (via the framework's `param_patch` + `constant_replace`
edit kind). Everything else here is a fixed search baseline:

  - Qwen3-0.6B in bf16, full fine-tuning (no LoRA / no PEFT).
  - Manual training loop with a weighted multi-source sampler, AdamW,
    grad-ckpt + flash_attention_2 if available (falls back to SDPA).
  - Train until either ``SFT_TIME_BUDGET_SECONDS`` (default 600 = 10 min)
    of wall time has elapsed, or 5,000 steps have run, whichever comes first.
  - Periodic balanced val every ``EVAL_EVERY_STEPS`` (default 200) so wandb
    shows val curves; final ``val_loss`` is the *balanced* mean cross-entropy
    across all five buckets and is the canonical metric written to metrics.json.

Outputs ``metrics.json`` with at minimum ``val_loss`` (finite float).
The framework's result-validator reads this; the manager optimizes it.

Run:
    SFT_TIME_BUDGET_SECONDS=600 \\
    QWEN_SFT_METRICS_PATH=/abs/path/metrics.json \\
    python train.py
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

# wandb is optional; if missing or WANDB_DISABLED=1, training continues silently.
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


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
MODEL_NAME     = "Qwen/Qwen3-0.6B"
SEQ_LEN        = 2048
MICRO_BATCH    = 16
GRAD_ACCUM     = 2
LEARNING_RATE  = 2e-5
WARMUP_STEPS   = 50
WEIGHT_DECAY   = 0.01
GRAD_CKPT      = True
MAX_STEPS      = 5000
LOG_EVERY      = 10
EVAL_EVERY_STEPS = int(os.environ.get("EVAL_EVERY_STEPS", "200"))

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
# wandb glue
# ---------------------------------------------------------------------------

def maybe_init_wandb(config):
    """Return an active wandb run or None if wandb is unavailable / disabled."""
    if not _WANDB_AVAILABLE:
        return None
    if os.environ.get("WANDB_DISABLED") == "1":
        return None
    project = os.environ.get("WANDB_PROJECT", "workshop")
    entity = os.environ.get("WANDB_ENTITY", "haolong")
    name = os.environ.get("WANDB_RUN_NAME") or None
    group = os.environ.get("WANDB_RUN_GROUP") or None
    tags_env = os.environ.get("WANDB_TAGS", "qwen-sft,datamix,framework")
    tags = [t.strip() for t in tags_env.split(",") if t.strip()]
    try:
        return wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            tags=tags,
            config=config,
            reinit=True,
        )
    except Exception as e:
        print(f"wandb.init failed: {type(e).__name__}: {e}; continuing without wandb.")
        return None


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
        excess = len(full_ids) - seq_len
        full_ids = full_ids[excess:]
        prompt_ids = prompt_ids[excess:] if len(prompt_ids) > excess else []

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = list(full_ids)
    labels = list(full_ids)
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
        self._cache = {b: [None] * len(self.samples[b]) for b in self.buckets}
        for b, lst in self.samples.items():
            if not lst:
                raise FileNotFoundError(
                    f"bucket {b!r} {split} split is empty at {data_root}/{b}/{split}.jsonl. "
                    f"Run prepare.py first.")
        _seed = int(os.environ.get("QWEN_SFT_SEED", "42"))
        self.rng = np.random.default_rng(seed=_seed)

    def _tokenize_one(self, bucket, idx):
        cached = self._cache[bucket][idx]
        if cached is not None:
            return cached
        sample = self.samples[bucket][idx]
        ids = render_with_assistant_loss_mask(sample["messages"], self.tokenizer, self.seq_len)
        if ids is None:
            ids = (None, None)
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


def balanced_val_per_bucket(buckets, data_root, tokenizer, seq_len, max_per_bucket=200):
    """Build per-bucket held-out val tensors. Returns dict: bucket -> (xs, ys).

    Each bucket contributes up to ``max_per_bucket`` samples; they stay split
    by bucket so we can report per-bucket val_loss in addition to overall.
    Tensors are CPU-resident; the eval loop moves micro-batches to DEVICE.
    """
    out = {}
    for bucket in buckets:
        bucket_path = Path(data_root) / bucket / "val.jsonl"
        if not bucket_path.exists():
            continue
        samples = load_bucket_jsonl(bucket_path)[:max_per_bucket]
        xs, ys = [], []
        for s in samples:
            ids = render_with_assistant_loss_mask(s["messages"], tokenizer, seq_len)
            if ids is None:
                continue
            xs.append(ids[0])
            ys.append(ids[1])
        if xs:
            out[bucket] = (torch.tensor(xs, dtype=torch.long),
                           torch.tensor(ys, dtype=torch.long))
    if not out:
        raise RuntimeError(f"No valid val samples found under {data_root}.")
    return out


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_balanced(model, val_data_per_bucket, eval_bs):
    """Compute per-bucket and overall token-weighted val cross-entropy.

    Returns ``(overall_loss, per_bucket_loss_dict)``. Overall loss is the
    token-weighted mean across all buckets — the gameable-resistant metric
    the framework optimizes.
    """
    model.eval()
    overall_loss_sum = 0.0
    overall_token_count = 0
    per_bucket = {}
    for bucket, (xs, ys) in val_data_per_bucket.items():
        bucket_loss_sum = 0.0
        bucket_tokens = 0
        for i in range(0, xs.shape[0], eval_bs):
            xb = xs[i:i+eval_bs].to(DEVICE)
            yb = ys[i:i+eval_bs].to(DEVICE)
            logits = model(input_ids=xb).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = yb[:, 1:].contiguous()
            losses = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            )
            mask = shift_labels != -100
            bucket_loss_sum += losses.sum().item()
            bucket_tokens += int(mask.sum().item())
        per_bucket[bucket] = bucket_loss_sum / max(1, bucket_tokens)
        overall_loss_sum += bucket_loss_sum
        overall_token_count += bucket_tokens
    model.train()
    overall = overall_loss_sum / max(1, overall_token_count)
    return overall, per_bucket


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading tokenizer + model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    normalized_mix = {k: round(v / sum(DATA_MIX.values()), 4) for k, v in DATA_MIX.items()}
    print(f"params: total={num_params/1e9:.2f}B trainable={num_trainable/1e9:.2f}B")
    print(f"DATA_MIX (normalized): {normalized_mix}")

    print("building train sampler…")
    sampler = WeightedBucketSampler(DATA_MIX, DATA_ROOT, tokenizer, SEQ_LEN, split="train")
    print("building per-bucket val tensors…")
    val_data_per_bucket = balanced_val_per_bucket(list(DATA_MIX.keys()), DATA_ROOT, tokenizer, SEQ_LEN)
    val_total = sum(xs.shape[0] for xs, _ in val_data_per_bucket.values())
    print(f"val: {val_total} samples across {len(val_data_per_bucket)} buckets")

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

    # ---- wandb ----
    wandb_config = {
        "model_name":          MODEL_NAME,
        "seq_len":             SEQ_LEN,
        "micro_batch":         MICRO_BATCH,
        "grad_accum":          GRAD_ACCUM,
        "effective_batch":     MICRO_BATCH * GRAD_ACCUM,
        "learning_rate":       LEARNING_RATE,
        "warmup_steps":        WARMUP_STEPS,
        "weight_decay":        WEIGHT_DECAY,
        "grad_ckpt":           GRAD_CKPT,
        "max_steps":           MAX_STEPS,
        "time_budget_seconds": TIME_BUDGET,
        "eval_every_steps":    EVAL_EVERY_STEPS,
        "data_mix":            dict(DATA_MIX),
        "data_mix_normalized": normalized_mix,
        "attn_implementation": attn_impl,
        "num_params":          int(num_params),
        "num_params_trainable":int(num_trainable),
    }
    run = maybe_init_wandb(wandb_config)
    if run is not None:
        print(f"wandb run: {run.url}")

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

            do_eval = (EVAL_EVERY_STEPS > 0 and step % EVAL_EVERY_STEPS == 0)
            do_log = (step % LOG_EVERY == 0)

            if do_log or do_eval:
                avg = train_loss_running / max(1, train_loss_count)
                now = time.time()
                dt = now - last_log
                last_log = now
                cur_lr = lr_at(step)
                log_payload = {
                    "train/loss":   avg,
                    "train/lr":     cur_lr,
                    "elapsed":      elapsed,
                    "ms_per_step":  dt * 1000 / max(1, LOG_EVERY),
                }
                print(f"step {step:05d} ({elapsed:.0f}s/{TIME_BUDGET}s) | "
                      f"loss: {avg:.4f} | lr: {cur_lr:.2e} | "
                      f"dt: {dt*1000/max(1, LOG_EVERY):.0f}ms",
                      flush=True)
                train_loss_running = 0.0
                train_loss_count = 0

                if do_eval:
                    val_loss, per_bucket = evaluate_balanced(model, val_data_per_bucket, MICRO_BATCH)
                    log_payload["val/overall"] = val_loss
                    for b, l in per_bucket.items():
                        log_payload[f"val/{b}"] = l
                    print(f"step {step:05d} | val/overall: {val_loss:.4f} | "
                          f"per_bucket: {{{', '.join(f'{b}: {l:.4f}' for b, l in per_bucket.items())}}}",
                          flush=True)

                if run is not None:
                    wandb.log(log_payload, step=step)

    train_seconds = time.time() - t_start
    print(f"training done after {step} steps / {train_seconds:.0f}s")

    # ----------------- final balanced val loss (canonical metric) -----------------
    print("evaluating final balanced val…")
    final_val_loss, final_per_bucket = evaluate_balanced(model, val_data_per_bucket, MICRO_BATCH)
    val_token_count = sum(xs.shape[0] * (xs.shape[1] - 1) for xs, _ in val_data_per_bucket.values())
    print(f"val_loss (final): {final_val_loss:.4f}")
    for b, l in final_per_bucket.items():
        print(f"  {b:<10s} {l:.4f}")

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    if run is not None:
        wandb.log({
            "val/overall_final": final_val_loss,
            **{f"val/{b}_final": l for b, l in final_per_bucket.items()},
        }, step=step)
        wandb.summary["val_loss"] = final_val_loss
        wandb.summary["training_seconds"] = train_seconds
        wandb.summary["num_steps"] = step
        wandb.summary["peak_vram_mb"] = peak_vram_mb
        for b, l in final_per_bucket.items():
            wandb.summary[f"val_{b}"] = l
        wandb.finish()

    write_metrics({
        "val_loss":             float(final_val_loss),
        "val_loss_per_bucket":  {b: float(l) for b, l in final_per_bucket.items()},
        "training_seconds":     float(train_seconds),
        "total_seconds":        float(time.time() - t_start),
        "num_steps":            int(step),
        "num_params":           int(num_params),
        "num_params_trainable": int(num_trainable),
        "peak_vram_mb":         float(peak_vram_mb),
        "data_mix":             dict(DATA_MIX),
        "data_mix_normalized":  normalized_mix,
        "model_name":           MODEL_NAME,
        "seq_len":              int(SEQ_LEN),
        "micro_batch":          int(MICRO_BATCH),
        "grad_accum":           int(GRAD_ACCUM),
        "effective_batch":      int(MICRO_BATCH * GRAD_ACCUM),
        "learning_rate":        float(LEARNING_RATE),
        "warmup_steps":         int(WARMUP_STEPS),
        "weight_decay":         float(WEIGHT_DECAY),
        "time_budget_seconds":  int(TIME_BUDGET),
        "eval_every_steps":     int(EVAL_EVERY_STEPS),
        "device_name":          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    })


if __name__ == "__main__":
    main()
