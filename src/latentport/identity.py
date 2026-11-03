"""Exact, inspectable identities for models and inference states."""
from __future__ import annotations

from dataclasses import dataclass

from .errors import CompatibilityError


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    repo_id: str
    revision: str
    backend: str
    weights: str
    tokenizer_revision: str
    template_sha256: str
    state_abi: str

    def __post_init__(self):
        if not all((self.repo_id, self.revision, self.backend, self.weights,
                    self.tokenizer_revision, self.template_sha256, self.state_abi)):
            raise ValueError("A state identity must be fully specified")

    def require_same(self, other: "ModelIdentity") -> None:
        if self != other:
            raise CompatibilityError(f"State is bound to {self.repo_id}@{self.revision} "
                                     f"({self.backend}/{self.weights}/{self.state_abi}), not "
                                     f"{other.repo_id}@{other.revision} "
                                     f"({other.backend}/{other.weights}/{other.state_abi})")


@dataclass(frozen=True, slots=True)
class StateInfo:
    identity: ModelIdentity
    clock: int
    phase: str
    origin: str
    target_prompt_tokens: int
    representation: str
    tensor_count: int
    tensor_bytes: int
    shares_storage: bool
    state_id: str
    parent_id: str | None


@dataclass(frozen=True, slots=True)
class RunStats:
    queue_s: float = 0.0
    reader_prefill_s: float = 0.0
    reader_sync_s: float = 0.0
    target_prefill_s: float = 0.0
    translation_s: float = 0.0
    device_transfer_s: float = 0.0
    state_install_s: float = 0.0
    target_first_token_compute_s: float = 0.0
    first_token_s: float = 0.0
    request_ttft_s: float = 0.0
    generation_s: float = 0.0
    total_s: float = 0.0
    peak_vram_bytes: int | None = None
    target_prompt_tokens: int = 0
    generated_tokens: int = 0
    backend: str = ""
    mtp_used: bool = False
    mtp_requested: bool = False
    mtp_proposed_tokens: int = 0
    mtp_accepted_tokens: int = 0
    mtp_warmup_tokens: int = 0
    mtp_batches: int = 0


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    token_ids: tuple[int, ...]
    state: "State"
    finished: bool
    stats: RunStats


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .state import State
