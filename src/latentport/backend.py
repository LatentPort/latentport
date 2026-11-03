"""Small backend contract: complete state in, complete state out."""
from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence


class Backend(Protocol):
    name: str
    state_abi: str
    supports_mtp: bool

    def codec(self, identity: "ModelIdentity") -> "Codec": ...

    def capture(self, identity: "ModelIdentity", token_ids: Sequence[int]) -> Mapping[str, Any]: ...

    def advance(self, identity: "ModelIdentity", components: Mapping[str, Any],
                clock: int, token_ids: Sequence[int]) -> Mapping[str, Any]: ...

    def first_logits(self, identity: "ModelIdentity", components: Mapping[str, Any],
                     clock: int) -> Any: ...

    def eos_ids(self, identity: "ModelIdentity") -> frozenset[int]: ...

    def install(self, identity: "ModelIdentity", components: Mapping[str, Any],
                clock: int) -> None: ...

    def begin_decode(self, identity: "ModelIdentity", components: Mapping[str, Any],
                     clock: int, *, mtp: bool = False) -> "DecodeHandle": ...


class DecodeHandle(Protocol):
    @property
    def logits(self) -> Any: ...

    @property
    def clock(self) -> int: ...

    def step(self, token_id: int) -> None: ...

    def snapshot(self) -> Mapping[str, Any]: ...


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .codec import Codec
    from .identity import ModelIdentity
