# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SecureKV M2 end-to-end test (offline vLLM engine, small model).

Three phases, each in its own process (driver: e2e_driver.sh):

  baseline  no connector           -> ground-truth greedy output
  store     SecureKVConnector      -> same prompt; KV sealed to disk
  load      SecureKVConnector, NEW  -> external encrypted hit; KV decrypted
            process                   and injected

Acceptance: all three outputs are byte-identical (greedy + byte-exact KV
round-trip make this a hard equivalence check), and everything on disk is
sealed (magic + high entropy).

Usage:
  SKV_MASTER_KEY=<hex> python e2e_test.py --phase {baseline|store|load} \
      --model <path> --storage <dir> --out <result.json>
"""

import argparse
import json
import os


PROMPT = (
    "You are a technical writer. Explain, in clear English prose, how a "
    "paged key-value cache works in a large language model inference "
    "engine. Cover the following aspects one by one: why contiguous "
    "pre-allocation wastes memory, how fixed-size blocks solve external "
    "fragmentation, what a block table is, how copy-on-write enables "
    "sharing between sequences that have a common prefix, and why block "
    "granularity matters for prefix caching hit rates. Then give a short "
    "numeric example with a block size of sixteen tokens and a prompt of "
    "three hundred tokens, computing how many blocks are needed and how "
    "many token slots remain unused in the final block. Keep the tone "
    "factual and avoid marketing language. Begin your explanation now:"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["baseline", "store", "load"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--storage", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    kwargs = {}
    if args.phase != "baseline":
        kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="SecureKVConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "backend": "local_disk",
                "backend_config": {"root_dir": args.storage},
                "key_source": "env:SKV_MASTER_KEY",
                "tenant_id": "default",
            },
        )

    llm = LLM(
        model=args.model,
        max_model_len=512,
        gpu_memory_utilization=0.30,
        enforce_eager=True,
        max_num_seqs=2,
        disable_hybrid_kv_cache_manager=True,
        enable_prefix_caching=False,  # isolate the external-cache path
        **kwargs,
    )
    out = llm.generate(
        [PROMPT],
        SamplingParams(temperature=0.0, max_tokens=48, seed=42),
    )[0]

    result = {
        "phase": args.phase,
        "prompt_tokens": len(out.prompt_token_ids),
        "text": out.outputs[0].text,
        "token_ids": list(out.outputs[0].token_ids),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"[{args.phase}] {result['prompt_tokens']} prompt toks -> "
          f"{len(result['token_ids'])} out toks | {result['text'][:70]!r}")


if __name__ == "__main__":
    main()
