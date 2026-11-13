"""Pinned, tensor-only LPCompact selector; loaded only for an actual compaction."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

from ..errors import CompatibilityError

@lru_cache(maxsize=4)
def _model(device):
    import torch
    from safetensors.torch import load_file
    spec = json.loads((Path(__file__).parent.parent / "configs/lpcompact.json").read_text())
    override = os.environ.get("LATENTPORT_COMPACTOR")
    if override:
        path = Path(override).expanduser()
    else:
        from huggingface_hub import hf_hub_download
        from ..hub import bulk_home
        path = Path(hf_hub_download(spec["repo_id"], spec["filename"], revision=spec["revision"],
                                    cache_dir=bulk_home() / "huggingface"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise CompatibilityError("LPCompact selector checksum mismatch")
    # Loading an operation must not perturb the caller's sampling RNG.
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Sequential(torch.nn.Linear(3072, 128), torch.nn.GELU(), torch.nn.Linear(128, 1))
    model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    return model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)


def score(values):
    import torch
    device = values["V.3"].device
    model = _model(str(device))
    chunks = []
    # Bounded scratch space at 128K and beyond; math matches the frozen selector.
    for start in range(0, values["V.3"].shape[2], 8192):
        x = torch.cat([values[f"V.{i}"][0, :, start:start + 8192].transpose(0, 1).reshape(-1, 1024)
                       for i in (3, 15, 31)], dim=-1).float()
        x = x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
        chunks.append(model(x).squeeze(-1))
    return torch.cat(chunks).cpu().tolist()
