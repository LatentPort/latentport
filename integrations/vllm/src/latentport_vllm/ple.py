"""Read-only mapping for BF16 PLE shards consumed by vLLM's disk offloader."""

import json
import mmap

import torch
from safetensors import safe_open


def readonly_ple_shard(path):
    """Return file-backed views for an embedding-only shard, or defer to vLLM.

    The disk offloader only records each view's file and byte offset. PyTorch's
    default writable private mapping can reserve the size of the entire shard
    against host RAM even though the offloader never makes a private copy.
    """
    # Validate the format and inspect tensor metadata without opening a writable
    # PyTorch mapping or materializing any tensor.
    with safe_open(path, framework="numpy") as handle:
        names = handle.keys()
        if not names or any(
            ".ngram_embedding." not in name
            or handle.get_slice(name).get_dtype() != "BF16"
            for name in names
        ):
            return None
    with open(path, "rb") as file:
        header_size = int.from_bytes(file.read(8), "little")
        header = json.loads(file.read(header_size))
        mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)

    tensors = []
    for name in names:
        record = header[name]
        start, stop = record["data_offsets"]
        view = torch.frombuffer(
            mapping,
            dtype=torch.bfloat16,
            count=(stop - start) // 2,
            offset=8 + header_size + start,
        )
        tensors.append((name, view.reshape(record["shape"])))
    return tensors
