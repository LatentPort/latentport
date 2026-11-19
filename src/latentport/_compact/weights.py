"""Pinned, tensor-only LPCompact selector; loaded only for an actual compaction."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

from ..errors import CompatibilityError
from .._models import DEFAULT_MODEL, MODEL_REVISIONS, feature_layers


def selector_config(repo):
    if repo not in MODEL_REVISIONS:
        raise CompatibilityError(f"Unsupported LPCompact base: {repo}")
    name = "lpcompact" if repo == DEFAULT_MODEL else "lpcompact-" + repo.split("/")[-1].lower()
    return json.loads((Path(__file__).parent.parent / "configs" / (name + ".json")).read_text())

@lru_cache(maxsize=8)
def _model(device, repo, override):
    import torch
    from safetensors.torch import load_file
    spec = selector_config(repo)
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
        width, hidden = spec["feature_dimension"], spec["hidden_dimension"]
        model = torch.nn.Sequential(torch.nn.Linear(width, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 1))
    model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    return model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)


def score_features(model, features):
    import torch

    x = torch.cat([feature.flatten(1) for feature in features], dim=-1).float()
    x = x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    return model(x).squeeze(-1)


def score(values, repo=DEFAULT_MODEL):
    import torch
    layers = feature_layers(repo)
    reference = values[f"V.{layers[0]}"]
    device = reference.device
    model = _model(str(device), repo, os.environ.get("LATENTPORT_COMPACTOR"))
    chunks = []
    # Bounded scratch space at 128K and beyond; math matches the frozen selector.
    for start in range(0, reference.shape[2], 8192):
        features = [values[f"V.{i}"][0, :, start:start + 8192].transpose(0, 1).to(device)
                    for i in layers]
        chunks.append(score_features(model, features))
    return torch.cat(chunks).cpu().tolist()
