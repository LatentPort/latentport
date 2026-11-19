"""Pinned GGUF counterparts for the base checkpoints in the selector catalog."""

from dataclasses import dataclass
from hashlib import file_digest
from pathlib import Path

from latentport._models import DEFAULT_MODEL, FLASH_NEXT, GEMMA, QWEN3, QWEN36


@dataclass(frozen=True)
class GGUFCheckpoint:
    repo: str
    revision: str
    # Filename and Hugging Face LFS SHA-256 for each part, in shard order.
    files: tuple[tuple[str, str], ...]


CHECKPOINTS = {
    "Qwen/Qwen3.5-4B": GGUFCheckpoint(
        "lmstudio-community/Qwen3.5-4B-GGUF",
        "f9f88ac3e234be915e23811a6d28ea287bdb927e",
        (
            (
                "Qwen3.5-4B-Q4_K_M.gguf",
                "25082a7dd3776cc3c741c6347d3bd04523f05796607b3fbc32fa3a25dfa1418c",
            ),
        ),
    ),
    "Qwen/Qwen3.5-9B": GGUFCheckpoint(
        "lmstudio-community/Qwen3.5-9B-GGUF",
        "1379f25c6b505a3fc737bd7818cb09389cf807c1",
        (
            (
                "Qwen3.5-9B-Q4_K_M.gguf",
                "cd76ec205963b3b33350093e6904d9de16c4e666fd104e1f632d25c7f15f2a13",
            ),
        ),
    ),
    QWEN3: GGUFCheckpoint(
        "Qwen/Qwen3-14B-GGUF",
        "530227a7d994db8eca5ab5ced2fb692b614357fd",
        (
            (
                "Qwen3-14B-Q4_K_M.gguf",
                "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0",
            ),
        ),
    ),
    QWEN36: GGUFCheckpoint(
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "a483e9e6cbd595906af30beda3187c2663a1118c",
        (
            (
                "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
                "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61",
            ),
        ),
    ),
    DEFAULT_MODEL: GGUFCheckpoint(
        "bartowski/Qwen3.8-27B-GGUF",
        "87410e7b9a5c21e6ca70c0c9cc5237de909fe621",
        (
            (
                "Qwen3.8-27B-Q4_K_M.gguf",
                "880a52af8317974f7b4b064a6bc29ff49e188b91aa218e08eea3078b6b1cc673",
            ),
        ),
    ),
    FLASH_NEXT: GGUFCheckpoint(
        "ggml-org/Qwen3.8-Flash-Next-GGUF",
        "a3e717e66206086bffa2be26abbbf22e84e93f36",
        (
            (
                "Qwen3.8-Flash-Next-Q2_K-00001-of-00002.gguf",
                "c6889865707a9a7ab1da0d177b7b0751e8c8d1fb1dae1f8a81577b7e2d23d550",
            ),
            (
                "Qwen3.8-Flash-Next-Q2_K-00002-of-00002.gguf",
                "75d1ead63e5f911b5ea6b5dc35280af8707772cc490991c528c1df5301636420",
            ),
        ),
    ),
    GEMMA: GGUFCheckpoint(
        "unsloth/gemma-4-31B-it-GGUF",
        "c1ac76e99d5513b141e8adde7288b85c3f9c32ec",
        (
            (
                "gemma-4-31B-it-Q4_K_M.gguf",
                "38bd64c852c4b460434cc7162fa9bdcf242faf86502581a754cb72956bb17f84",
            ),
        ),
    ),
}


def gguf_files(base_model, local_path, cache_dir):
    from huggingface_hub import hf_hub_download

    checkpoint = CHECKPOINTS[base_model]
    if local_path is None:
        return tuple(
            hf_hub_download(
                checkpoint.repo,
                filename,
                revision=checkpoint.revision,
                cache_dir=cache_dir,
            )
            for filename, _ in checkpoint.files
        )

    first = Path(local_path).expanduser().absolute()
    if first.name != checkpoint.files[0][0]:
        raise ValueError(f"--gguf-path must name {checkpoint.files[0][0]}")
    paths = tuple(first.with_name(filename) for filename, _ in checkpoint.files)
    for path, (_, digest) in zip(paths, checkpoint.files, strict=True):
        with path.open("rb") as file:
            if file_digest(file, "sha256").hexdigest() != digest:
                raise ValueError(
                    f"GGUF file does not match the pinned checkpoint: {path}"
                )
    return tuple(map(str, paths))
