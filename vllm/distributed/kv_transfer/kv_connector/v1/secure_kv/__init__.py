# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SecureKV: encrypted external KV cache connector (AES-256-GCM at the
GPU-egress boundary). See kv-encryption-connector-design.md."""

from vllm.distributed.kv_transfer.kv_connector.v1.secure_kv.connector import (
    SecureKVConnector,
)

__all__ = ["SecureKVConnector"]
