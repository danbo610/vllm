# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SynthID-Text watermark detector (mean-g z-test, CPU only).

Companion tool for the builtin ``SynthIDLogitsProcessor``
(``vllm/v1/sample/logits_processor/synthid.py``). Detection needs ONLY the
tokenizer plus the same key/config as generation — no GPU, no vLLM server.

Usage::

    python detect.py --tokenizer /path/to/model --text "..."
    python detect.py --tokenizer /path/to/model --jsonl results.jsonl

The tokenizer path may also be set via the ``SYNTHID_TOKENIZER`` env var.
Keys / ngram length come from ``SYNTHID_KEYS`` / ``SYNTHID_NGRAM_LEN`` and
MUST match the serving side exactly.

z-score interpretation: unwatermarked text ~ N(0, 1); watermarked >> 0
(z > 4 => p < 3e-5, strong evidence).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
from transformers import AutoTokenizer
from transformers.generation import SynthIDTextWatermarkLogitsProcessor

# Must match vllm/v1/sample/logits_processor/synthid.py exactly.
_ENV_KEYS = os.environ.get("SYNTHID_KEYS", "")
SYNTHID_KEYS = (
    [int(x) for x in _ENV_KEYS.split(",") if x.strip()]
    if _ENV_KEYS
    else [654, 400, 836, 123, 340, 443, 597, 160, 57, 29,
          590, 639, 13, 715, 468, 990, 966, 226, 324, 585]
)
NGRAM_LEN = int(os.environ.get("SYNTHID_NGRAM_LEN", "5"))


def build_processor(device: str = "cpu") -> SynthIDTextWatermarkLogitsProcessor:
    """CPU processor for detection. Generating the sampling table on CPU is
    what makes it match the serving side, which shares a CPU-canonical table
    (same seed on CUDA would yield a different table)."""
    return SynthIDTextWatermarkLogitsProcessor(
        ngram_len=NGRAM_LEN,
        keys=SYNTHID_KEYS,
        sampling_table_size=65536,
        sampling_table_seed=0,
        context_history_size=1024,
        device=torch.device(device),
    )


def score_ids(ids: torch.Tensor, proc) -> dict | None:
    """ids: [1, seq] token ids of the *generated* text."""
    if ids.shape[1] < NGRAM_LEN + 4:
        return None
    g = proc.compute_g_values(ids)                       # [1, T, depth]
    mask = proc.compute_context_repetition_mask(ids)     # [1, T]
    n = int(mask.sum().item()) * g.shape[-1]
    if n == 0:
        return None
    mean_g = float((g * mask.unsqueeze(-1)).sum().item()) / n
    z = (mean_g - 0.5) / math.sqrt(0.25 / n)
    return {"tokens": int(ids.shape[1]), "n_g": n,
            "mean_g": round(mean_g, 4), "z": round(z, 2)}


def score_text(text: str, tok, proc) -> dict | None:
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    return score_ids(ids, proc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default=os.environ.get("SYNTHID_TOKENIZER"),
                    help="HF tokenizer path/name (env: SYNTHID_TOKENIZER)")
    ap.add_argument("--text", help="score a single text")
    ap.add_argument("--jsonl", help='score {"text", "group"} lines')
    args = ap.parse_args()

    if not args.tokenizer:
        sys.exit("error: --tokenizer (or SYNTHID_TOKENIZER env) is required")
    if not args.text and not args.jsonl:
        sys.exit("error: one of --text / --jsonl is required")

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    proc = build_processor()

    if args.text:
        print(json.dumps(score_text(args.text, tok, proc), ensure_ascii=False))
        return

    groups: dict[str, list[float]] = {}
    with open(args.jsonl) as f:
        for line in f:
            rec = json.loads(line)
            s = score_text(rec["text"], tok, proc)
            if s is None:
                continue
            g = rec.get("group", "?")
            groups.setdefault(g, []).append(s["z"])
            print(f'[{g:>12}] tokens={s["tokens"]:>4} n_g={s["n_g"]:>5} '
                  f'mean_g={s["mean_g"]:.4f} z={s["z"]:>7.2f}  '
                  f'| {rec["text"][:40]!r}')
    print("\n=== summary (z-score) ===")
    for g, zs in sorted(groups.items()):
        zs_sorted = sorted(zs)
        print(f"{g:>12}: n={len(zs)} mean={sum(zs) / len(zs):6.2f} "
              f"min={zs_sorted[0]:6.2f} max={zs_sorted[-1]:6.2f}")


if __name__ == "__main__":
    main()
