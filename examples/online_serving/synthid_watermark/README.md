# SynthID-Text Watermarking Tools

Companion tools for the builtin `SynthIDLogitsProcessor`
(`vllm/v1/sample/logits_processor/synthid.py`), which integrates Google
DeepMind's SynthID-Text watermark (Nature 2024, via the Hugging Face
`transformers` implementation) into the vLLM v1 sampler.

| File | Purpose | Needs |
|---|---|---|
| `detect.py` | Score text for the watermark (mean-g z-test) | tokenizer only — no GPU, no server |
| `ab_experiment.py` | 8-prompt watermark-on/off A/B run against a live server | running vLLM server |
| `selftest.py` | Offline check of the builtin processor + detector round-trip | vLLM installed, CPU only |

## Enabling the watermark (per request)

The processor is built in and loaded by default; watermarking is opt-in
per request via the OpenAI-compatible extension field `vllm_xargs`:

```json
{"model": "...", "messages": [...], "temperature": 1.0,
 "vllm_xargs": {"synthid_wm": 1}}
```

Requests without the flag take a zero-overhead path. **Standard clients
(Claude Code, plain OpenAI/Anthropic SDK calls) never send `vllm_xargs`,
so their traffic is NOT watermarked** unless a gateway injects the field.

## Detecting

```bash
python detect.py --tokenizer /path/to/served/model --text "…generated text…"
# {"tokens": 420, "n_g": 8000, "mean_g": 0.5381, "z": 6.82}

python detect.py --tokenizer /path/to/served/model --jsonl results.jsonl
```

Interpretation: unwatermarked text scores z ~ N(0, 1); watermarked text
scores far above (z > 4 ⇒ p < 3e-5). Longer texts give stronger scores;
below ~200 tokens the test loses power.

## A/B experiment

```bash
WM_MODEL=my-model WM_URL=http://localhost:8000/v1/chat/completions \
WM_KEY=my-key python ab_experiment.py
python detect.py --tokenizer /path/to/served/model --jsonl results.jsonl
```

Reference numbers from a Qwen3.8-27B deployment (TP=2, L20): watermarked
8/8 detected, z avg ≈ 8.8; plain controls z avg ≈ 0.1; ~12% latency on
opted-in requests only.

## Consistency requirements (important)

Generation and detection must use the **same configuration**, wired
through the same env vars on both sides:

| Env var | Meaning | Default |
|---|---|---|
| `SYNTHID_KEYS` | comma-separated watermark keys | demo keys — **replace in production** and keep secret |
| `SYNTHID_NGRAM_LEN` | context window length | 5 |
| `SYNTHID_TOKENIZER` | tokenizer for `detect.py` | — (or `--tokenizer`) |

A key/ngram mismatch, or a different tokenizer, silently yields z ≈ 0.

## Known constraints

- Watermarking requires stochastic sampling: `temperature=0` disables it;
  small `top_k` weakens it.
- Async scheduling: the processor declares `needs_output_token_ids()`, so
  the engine repairs the `-1` placeholders before it runs (see
  vllm-project/vllm#52461). On engines without that mechanism, run the
  server with `--no-async-scheduling`.
- Not validated in combination with speculative decoding.
