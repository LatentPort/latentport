"""Frozen standalone base revisions and their complete-state geometries."""
from .context import PINNED_REVISION

DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
FLASH_NEXT = "Qwen/Qwen3.8-Flash-Next"
FLASH_ABI = "qwen4-exp.complete-indexed-text.v1"
QWEN3 = "Qwen/Qwen3-14B"
QWEN3_ABI = "qwen3.complete-attention-text.v1"
GEMMA = "google/gemma-4-31B-it"
GEMMA_ABI = "gemma4.sliding-global-text.v1"


def feature_layers(repo):
    return (5, 29, 59) if repo == GEMMA else (3, 15, 31)


MODEL_REVISIONS = {
    QWEN3: "40c069824f4251a91eefaf281ebe4c544efd3e18",
    GEMMA: "842da3794eaa0b77d5f08bae87a17459d91ff475",
    FLASH_NEXT: "812d96e10a0dd82b69d749e68693698cf6c36daa",
    DEFAULT_MODEL: PINNED_REVISION,
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
}


def model_geometry(repo):
    from ._reference.native import _geometry
    if repo == QWEN3:
        return dict(types=["full_attention"] * 40, hidden=5120, kv_heads=8, head_dim=128, minimum_context=32768)
    if repo == GEMMA:
        return dict(types=(["sliding_attention"] * 5 + ["full_attention"]) * 10,
                    hidden=5376, kv_heads=4, head_dim=512, sliding_window=1024,
                    sliding_heads=16, sliding_dim=256)
    if repo == FLASH_NEXT:
        return dict(types=(["linear_attention"] * 3 + ["full_attention"]) * 12,
                    hidden=2560, kv_heads=2, head_dim=256, recurrent_heads=48,
                    recurrent_k=128, recurrent_v=128, conv_channels=10240, conv_width=4,
                    index_width=128, ple_layer=1, ple_channels=10240, ple_width=9, ngram_width=2)
    if repo == DEFAULT_MODEL:
        return _geometry("target")
    if repo == "Qwen/Qwen3.5-4B":
        return _geometry("source")
    if repo == "Qwen/Qwen3.5-9B":
        # The pinned 4B and 9B configurations differ only in the terminal
        # hidden width among the fields in the complete-state geometry.
        return {**_geometry("source"), "hidden": 4096}
    raise KeyError(repo)
