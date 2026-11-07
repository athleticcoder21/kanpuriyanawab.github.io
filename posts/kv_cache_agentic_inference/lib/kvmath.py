"""Napkin model for KV cache, recurrent state, prefill and decode, derived from HF configs and checkpoints.

Notation follows the scaling book's inference chapter: L layers, K KV heads, N query heads,
H head dim, S context, T new tokens, B batch, C peak FLOP/s, W bandwidth (bytes/s).
"""

import json
import struct
import urllib.request
from collections import Counter
from dataclasses import dataclass
from functools import cache

KiB, MiB, GiB = 1024, 1024**2, 1024**3

DTYPE_BYTES = {"bfloat16": 2, "bf16": 2, "float16": 2, "float32": 4, "fp8_e4m3": 1, "fp8": 1, "nvfp4": 0.5}
# NVFP4 also stores one FP8 scale per 16 elements
NVFP4_BYTES = 0.5 + 1 / 16


@cache
def _get(url: str, rng: str | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng}"} if rng else {})
    with urllib.request.urlopen(req) as r:
        return r.read()


def hf_json(repo: str, name: str) -> dict:
    return json.loads(_get(f"https://huggingface.co/{repo}/raw/main/{name}"))


@cache
def param_counts(repo: str) -> Counter:
    """Parameters per category, read from every safetensors shard header."""
    counts = Counter()
    for shard in sorted(set(hf_json(repo, "model.safetensors.index.json")["weight_map"].values())):
        url = f"https://huggingface.co/{repo}/resolve/main/{shard}"
        n = struct.unpack("<Q", _get(url, "0-7"))[0]
        header = json.loads(_get(url, f"8-{7 + n}"))
        header.pop("__metadata__", None)
        for name, t in header.items():
            size = 1
            for s in t["shape"]:
                size *= s
            cat = ("visual" if "visual" in name else "mtp" if "mtp" in name else "embed" if "embed_tokens" in name
                   else "lm_head" if "lm_head" in name else "layers")
            counts[cat] += size
    return counts


@dataclass(frozen=True, eq=False)  # identity hash: one instance per model, used as a dict key
class Model:
    name: str
    params: Counter            # embed, layers, lm_head, visual, mtp
    attn_layers: int           # layers that keep a per-token KV cache
    q_heads: int               # N
    kv_heads: int              # K
    head_dim: int              # H
    state_bytes_per_seq: int   # recurrent (SSM + conv) state; 0 for pure attention

    # --- memory ---------------------------------------------------------------------------
    def kv_bytes_per_token(self, kv_dtype: str = "bf16") -> float:
        per_elem = NVFP4_BYTES if kv_dtype == "nvfp4" else DTYPE_BYTES[kv_dtype]
        return 2 * self.attn_layers * self.kv_heads * self.head_dim * per_elem

    def bytes_per_session(self, context: float, kv_dtype: str = "bf16") -> float:
        return context * self.kv_bytes_per_token(kv_dtype) + self.state_bytes_per_seq

    def resident_sessions(self, context: float, pool_bytes: float, kv_dtype: str = "bf16") -> float:
        return pool_bytes / self.bytes_per_session(context, kv_dtype)

    @property
    def served_weight_bytes(self) -> float:
        """Text-only serving loads embeddings, layers and LM head in BF16 (no vision tower, no MTP)."""
        return 2 * (self.params["embed"] + self.params["layers"] + self.params["lm_head"])

    # --- compute and traffic --------------------------------------------------------------
    @property
    def matmul_params(self) -> int:
        """Parameters every token multiplies through (the embedding is a lookup, not a matmul)."""
        return self.params["layers"] + self.params["lm_head"]

    def prefill_flops(self, T: float, S0: float = 0) -> float:
        """FLOPs to prefill T new tokens on top of S0 cached ones.

        Linear: 2 per matmul parameter per token (lm_head runs on the last token only).
        Attention: QK^T and AV each cost 2*N*H FLOPs per (query, key) pair per layer;
        a causal block of T queries over S0 + T keys has T*S0 + T^2/2 pairs.
        """
        linear = 2 * self.params["layers"] * T + 2 * self.params["lm_head"]
        pairs = T * S0 + T * T / 2
        return linear + 4 * self.attn_layers * self.q_heads * self.head_dim * pairs

    def decode_bytes_per_step(self, B: int, S: float, kv_dtype: str = "bf16") -> float:
        """Every decode step reads the weights once, each sequence's KV, and reads + writes its state."""
        weights = 2 * self.matmul_params
        return weights + B * (S * self.kv_bytes_per_token(kv_dtype) + 2 * self.state_bytes_per_seq)

    def decode_step_seconds(self, B: int, S: float, C: float, W: float, kv_dtype: str = "bf16") -> float:
        """Attention is bandwidth-bound; the matmuls are max(compute, weight load)."""
        kv = B * (S * self.kv_bytes_per_token(kv_dtype) + 2 * self.state_bytes_per_seq) / W
        mlp = max(2 * B * self.matmul_params / C, 2 * self.matmul_params / W)
        return kv + mlp


def load(repo: str) -> Model:
    cfg = hf_json(repo, "config.json")
    text = cfg.get("text_config", cfg)
    layer_types = text.get("layer_types") or ["full_attention"] * text["num_hidden_layers"]
    linear_layers = layer_types.count("linear_attention")

    state = 0
    if linear_layers:
        # Gated DeltaNet: one (k_dim x v_dim) matrix per value head, plus a short conv window
        ssm = (text["linear_num_value_heads"] * text["linear_key_head_dim"] * text["linear_value_head_dim"]
               * DTYPE_BYTES[text.get("mamba_ssm_dtype", "float32")])
        conv_dim = (2 * text["linear_num_key_heads"] * text["linear_key_head_dim"]
                    + text["linear_num_value_heads"] * text["linear_value_head_dim"])
        conv = conv_dim * (text["linear_conv_kernel_dim"] - 1) * DTYPE_BYTES[text["dtype"]]
        state = linear_layers * (ssm + conv)

    return Model(
        name=repo.split("/")[-1],
        params=param_counts(repo),
        attn_layers=layer_types.count("full_attention"),
        q_heads=text["num_attention_heads"],
        kv_heads=text["num_key_value_heads"],
        head_dim=text.get("head_dim", text["hidden_size"] // text["num_attention_heads"]),
        state_bytes_per_seq=state,
    )
