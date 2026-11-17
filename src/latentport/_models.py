"""Frozen standalone base revisions and their complete-state geometries."""
from .context import PINNED_REVISION

DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
MODEL_REVISIONS = {
    DEFAULT_MODEL: PINNED_REVISION,
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
}


def model_geometry(repo):
    from ._reference.native import _geometry
    if repo == DEFAULT_MODEL:
        return _geometry("target")
    if repo == "Qwen/Qwen3.5-4B":
        return _geometry("source")
    if repo == "Qwen/Qwen3.5-9B":
        # The pinned 4B and 9B configurations differ only in the terminal
        # hidden width among the fields in the complete-state geometry.
        return {**_geometry("source"), "hidden": 4096}
    raise KeyError(repo)
