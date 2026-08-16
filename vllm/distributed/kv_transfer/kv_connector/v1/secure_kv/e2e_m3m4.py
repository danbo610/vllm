# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SecureKV M3/M4 end-to-end: tenant keying + remote backend + parallel
decrypt, against a real vLLM engine. Driver: run_e2e_m3m4.sh.

Phases (fresh process each):
  store_acme   tenant=acme, backend=remote  -> KV sealed under acme key
  load_acme    tenant=acme, backend=remote  -> encrypted hit, decrypt, inject
  load_beta    tenant=beta, backend=remote  -> DIFFERENT index key => MISS,
                                               recompute (isolation proof)

Acceptance: acme load output == baseline; beta sees no external hit (its
HMAC index key differs), so it recomputes and still produces the baseline
output. All wired through the remote socket store holding only ciphertext.
"""

import argparse
import json
import os

PROMPT = (
    "You are a technical writer. Explain, in clear English prose, how a "
    "paged key-value cache works in a large language model inference "
    "engine. Cover why contiguous pre-allocation wastes memory, how "
    "fixed-size blocks solve external fragmentation, what a block table "
    "is, and how copy-on-write enables prefix sharing. Then give a short "
    "numeric example with block size sixteen and a three-hundred-token "
    "prompt. Keep the tone factual. Begin now:"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["baseline", "store_acme", "load_acme", "load_beta"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=14588)
    ap.add_argument("--backend", default="remote",
                    choices=["remote", "local_disk"])
    ap.add_argument("--disk-root", default="/tmp/skv_m3m4/disk")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    tenant = {"store_acme": "acme", "load_acme": "acme",
              "load_beta": "beta"}.get(args.phase)

    kwargs = {}
    if args.phase != "baseline":
        backend_config = ({"host": "127.0.0.1", "port": args.port}
                          if args.backend == "remote"
                          else {"root_dir": args.disk_root})
        kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="SecureKVConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "backend": args.backend,
                "backend_config": backend_config,
                "key_source": "env:SKV_MASTER_KEY",
                "tenant_id": tenant,   # set connector default tenant per phase
            },
        )

    llm = LLM(
        model=args.model,
        max_model_len=512,
        gpu_memory_utilization=0.30,
        enforce_eager=True,
        max_num_seqs=2,
        disable_hybrid_kv_cache_manager=True,
        enable_prefix_caching=False,
        **kwargs,
    )
    out = llm.generate([PROMPT],
                       SamplingParams(temperature=0.0, max_tokens=48, seed=42))[0]

    result = {"phase": args.phase, "tenant": tenant,
              "text": out.outputs[0].text,
              "token_ids": list(out.outputs[0].token_ids)}
    with open(args.out, "w") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"[{args.phase}] tenant={tenant} -> {len(result['token_ids'])} toks "
          f"| {result['text'][:60]!r}")


if __name__ == "__main__":
    main()
