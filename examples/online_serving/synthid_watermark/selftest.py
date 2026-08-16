# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline self-test for the builtin SynthIDLogitsProcessor (CPU only).

Drives the processor exactly the way vLLM does — a ``BatchUpdate`` whose
``output_tok_ids`` is a live list reference that grows each step — then
cross-checks the generated sequence with the detector from ``detect.py``.

No GPU and no server needed. If this passes but a live server shows z ~ 0,
the problem is in server-side data flow (extra_args delivery, placeholder
repair under async scheduling, call ordering), not in the processor math.

Run::

    python selftest.py
"""

import math

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate, SynthIDLogitsProcessor
from vllm.v1.sample.logits_processor.synthid import NGRAM_LEN  # noqa: F401

from detect import build_processor

VOCAB = 50000
STEPS = 300
SEED = 3


def run(watermark: bool) -> list[int]:
    proc = SynthIDLogitsProcessor(None, torch.device("cpu"), False)
    sp = SamplingParams(extra_args={"synthid_wm": 1} if watermark else None)
    out_ids: list[int] = []          # live reference, appended below
    prompt = list(range(100, 140))   # fake prompt
    bu = BatchUpdate(batch_size=1, removed=[], moved=[],
                     added=[(0, sp, prompt, out_ids)])
    proc.update_state(bu)
    print(f"watermark={watermark}: rows registered = {len(proc.rows)}")

    gen = torch.Generator().manual_seed(SEED)
    for _ in range(STEPS):
        logits = torch.randn(1, VOCAB, generator=gen) * 4.0
        logits = proc.apply(logits)
        probs = torch.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(probs, 1, generator=gen).item())
        out_ids.append(nxt)          # simulate vLLM appending the sampled token
    return list(out_ids)


def score(seq: list[int]) -> tuple[float, float]:
    det = build_processor()
    ids = torch.tensor([seq], dtype=torch.long)
    g = det.compute_g_values(ids)
    mask = det.compute_context_repetition_mask(ids)
    n = int(mask.sum().item()) * g.shape[-1]
    mean_g = float((g * mask.unsqueeze(-1)).sum().item()) / n
    return mean_g, (mean_g - 0.5) / math.sqrt(0.25 / n)


def main():
    wm = run(True)
    pl = run(False)
    wm_mean, wm_z = score(wm)
    pl_mean, pl_z = score(pl)
    print(f"watermarked: mean_g={wm_mean:.4f}  z={wm_z:7.2f}")
    print(f"plain      : mean_g={pl_mean:.4f}  z={pl_z:7.2f}")
    ok = wm_z > 6.0 and abs(pl_z) < 3.0
    print("SELFTEST:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
