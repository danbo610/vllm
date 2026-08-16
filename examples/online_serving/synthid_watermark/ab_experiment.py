# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SynthID watermark A/B experiment against a live vLLM server.

For each prompt, generate twice — watermark on (via ``vllm_xargs``) and
off — save both to JSONL, then score the file with ``detect.py``.

Configuration (env vars)::

    WM_URL         chat completions endpoint
                   (default http://localhost:8000/v1/chat/completions)
    WM_KEY         API key; empty = no Authorization header
    WM_MODEL       served model name (required)
    WM_MAX_TOKENS  per-request cap (default 420)
    WM_OUT         output JSONL path (default ./results.jsonl)

Then::

    python detect.py --tokenizer /path/to/model --jsonl results.jsonl
"""

from __future__ import annotations

import json
import os
import sys
import time

import requests

URL = os.environ.get("WM_URL", "http://localhost:8000/v1/chat/completions")
KEY = os.environ.get("WM_KEY", "")
MODEL = os.environ.get("WM_MODEL", "")
MAX_TOKENS = int(os.environ.get("WM_MAX_TOKENS", "420"))
OUT = os.environ.get("WM_OUT", "results.jsonl")

PROMPTS = [
    "写一段关于秋天的散文,大约三百字。",
    "Explain how a refrigerator works, in about 250 words.",
    "介绍一下围棋的基本规则和入门策略。",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "解释什么是复利,并举一个生活中的例子。",
    "Describe the water cycle for a middle-school student.",
    "写一封给十年后自己的信。",
    "Summarize the plot structure of a typical detective novel and why it works.",
]


def gen(prompt: str, watermark: bool) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        # The watermark needs stochastic sampling; temperature=0 would
        # disable it entirely.
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    # Explicit 1/0 keeps the A/B contrast correct in BOTH server modes:
    # opt-in (default) and SYNTHID_FORCE=1 (watermark-by-default).
    body["vllm_xargs"] = {"synthid_wm": 1 if watermark else 0}
    headers = {"Authorization": f"Bearer {KEY}"} if KEY else {}
    t0 = time.time()
    r = requests.post(URL, json=body, timeout=600, headers=headers)
    r.raise_for_status()
    d = r.json()
    return {
        "text": d["choices"][0]["message"]["content"],
        "secs": round(time.time() - t0, 1),
        "completion_tokens": d.get("usage", {}).get("completion_tokens"),
    }


def main():
    if not MODEL:
        sys.exit("error: WM_MODEL env var is required")
    records = []
    for i, p in enumerate(PROMPTS):
        for wm in (True, False):
            g = gen(p, wm)
            rec = {"group": "watermarked" if wm else "plain",
                   "prompt": p, **g}
            records.append(rec)
            print(f'[{i}] {"WM " if wm else "REF"} '
                  f'{g["completion_tokens"]} tok {g["secs"]}s | '
                  f'{g["text"][:50]!r}')
    with open(OUT, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nsaved {len(records)} records -> {OUT}")


if __name__ == "__main__":
    main()
