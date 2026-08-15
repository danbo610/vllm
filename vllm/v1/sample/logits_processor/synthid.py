# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SynthID-Text watermarking logits processor.

Wraps the Hugging Face ``transformers`` production implementation of
Google DeepMind's SynthID-Text watermark (Nature 2024) as a vLLM v1
stateful logits processor.

Opt-in per request via ``SamplingParams.extra_args`` (OpenAI API body field
``vllm_xargs``)::

    "vllm_xargs": {"synthid_wm": 1}

Configuration via environment variables (must match the detector side):

* ``SYNTHID_KEYS``: comma-separated watermark keys (defaults to demo keys;
  REPLACE with secret keys for any production use).
* ``SYNTHID_NGRAM_LEN``: watermark context window length (default 5).

Device-consistency note: the pseudo-random sampling table MUST be generated
on CPU. The same seed yields different tables on CPU vs CUDA RNGs, which
silently desynchronizes accelerator-side generation from CPU-side detection
("strong reweighting, zero detection"). The int64 hash arithmetic itself is
device-consistent once the table matches, so watermark compute stays on the
accelerator at negligible cost.

Behavioural notes:

* Requests without the opt-in flag are never touched (zero-overhead path).
* Watermarking requires stochastic sampling; greedy (``temperature=0``)
  requests receive no effective watermark.
* Not validated in combination with speculative decoding.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import BatchUpdate, LogitsProcessor

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)

_ENV_KEYS = os.environ.get("SYNTHID_KEYS", "")
SYNTHID_KEYS: list[int] = (
    [int(x) for x in _ENV_KEYS.split(",") if x.strip()]
    if _ENV_KEYS
    else [654, 400, 836, 123, 340, 443, 597, 160, 57, 29,
          590, 639, 13, 715, 468, 990, 966, 226, 324, 585]
)
NGRAM_LEN = int(os.environ.get("SYNTHID_NGRAM_LEN", "5"))
CONTEXT_HISTORY_SIZE = 1024
SAMPLING_TABLE_SEED = 0
SAMPLING_TABLE_SIZE = 65536

_canonical_table: torch.Tensor | None = None


def _get_canonical_table() -> torch.Tensor:
    """CPU-generated sampling table (see device-consistency note above)."""
    global _canonical_table
    if _canonical_table is None:
        from transformers.generation import SynthIDTextWatermarkLogitsProcessor

        tmp = SynthIDTextWatermarkLogitsProcessor(
            ngram_len=NGRAM_LEN,
            keys=SYNTHID_KEYS,
            sampling_table_size=SAMPLING_TABLE_SIZE,
            sampling_table_seed=SAMPLING_TABLE_SEED,
            context_history_size=CONTEXT_HISTORY_SIZE,
            device=torch.device("cpu"),
        )
        _canonical_table = tmp.sampling_table
    return _canonical_table


class _RowState:
    __slots__ = ("prompt_tail", "out_ids", "proc")

    def __init__(self, prompt_tail: list[int], out_ids: list[int],
                 proc) -> None:
        # Last ngram-1 prompt tokens (static seed for the sliding window).
        self.prompt_tail = prompt_tail
        # Live reference to the request's running output tokens (vLLM
        # guarantees this list is kept up to date; see BatchUpdate docs).
        self.out_ids = out_ids
        # Per-request HF processor: its internal context-repetition history
        # is thereby isolated per request, which a shared instance indexed
        # by batch row could not guarantee under continuous batching.
        self.proc = proc


class SynthIDLogitsProcessor(LogitsProcessor):
    """Per-request opt-in SynthID-Text watermarking."""

    def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                 is_pin_memory: bool) -> None:
        self.device = device
        self.rows: dict[int, _RowState] = {}

    def is_argmax_invariant(self) -> bool:
        # The watermark reweights logits and may change argmax.
        return False

    def _new_state(self, params: "SamplingParams",
                   prompt_tok_ids: list[int] | None,
                   output_tok_ids: list[int]) -> _RowState | None:
        extra = getattr(params, "extra_args", None) or {}
        if not extra.get("synthid_wm"):
            return None  # not opted in -> no state, zero overhead
        from transformers.generation import SynthIDTextWatermarkLogitsProcessor

        proc = SynthIDTextWatermarkLogitsProcessor(
            ngram_len=NGRAM_LEN,
            keys=SYNTHID_KEYS,
            sampling_table_size=SAMPLING_TABLE_SIZE,
            sampling_table_seed=SAMPLING_TABLE_SEED,
            context_history_size=CONTEXT_HISTORY_SIZE,
            device=self.device,
        )
        # Keep compute on the accelerator but share the CPU-canonical random
        # table with the (CPU-side) detector.
        proc.sampling_table = _get_canonical_table().to(self.device)
        tail = list(prompt_tok_ids[-(NGRAM_LEN - 1):]) if prompt_tok_ids else []
        return _RowState(tail, output_tok_ids, proc)

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        process_dict_updates(self.rows, batch_update, self._new_state)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.rows:
            return logits
        for idx, state in self.rows.items():
            ctx = (state.prompt_tail + state.out_ids)[-(NGRAM_LEN - 1):]
            if len(ctx) < NGRAM_LEN - 1:
                # Window not yet full (ultra-short prompt); skip this step.
                continue
            input_ids = torch.tensor([ctx], dtype=torch.long,
                                     device=logits.device)
            row = logits[idx].unsqueeze(0).float()
            logits[idx] = state.proc(input_ids, row).squeeze(0).to(logits.dtype)
        return logits
