"""Native-position Qwen3 execution through the shared model and state API."""

from dataclasses import dataclass

from .._models import QWEN3_ABI
from ..context import ContextOptions
from ..errors import CapabilityError
from ..state import State
from .gemma import GemmaBackend, GemmaDecode, GemmaLive
from .native import KERNEL_LOCK, ReferenceBackend
from .qwen3_decoder import Decoder


@dataclass(frozen=True, slots=True)
class Qwen3ContextOptions(ContextOptions):
    physical_limit: int = 40960

    native_positions: int = 40960

    @property
    def legacy(self):
        return False

    def record(self):
        return dict(
            position_policy="default",
            rope=dict(rope_type="default", rope_theta=1000000),
            logical_limit=self.logical_limit,
            physical_limit=self.physical_limit,
            execution="qwen3-native-text-positions.v1",
        )


class Qwen3Live(GemmaLive):
    @property
    def physical(self):
        return self.cache.get_seq_length(0)


class Qwen3Backend(GemmaBackend):
    state_abi = QWEN3_ABI
    codec = ReferenceBackend.codec

    def _model(self, identity):
        side = self._side(identity)
        with KERNEL_LOCK:
            if side not in self._models:
                self._models[side] = Decoder(self._path(side), self.device, options=self.options,
                                             nf4=self.name == "nf4")
        return self._models[side]

    def live(self, identity, state=None):
        return Qwen3Live(self, identity, state)

    def _run(self, identity, values, clock, ids):
        parent = (
            State(identity, clock, "context_ready", "install", _components=values)
            if values is not None
            else None
        )
        with Qwen3Live(self, identity, parent, _owned=True) as live:
            live.append(ids)
            return live.snapshot(release=True)._read_tensors()

    def eos_ids(self, identity):
        return frozenset((151643, 151645))

    def begin_decode(self, identity, components, clock, *, mtp=False):
        if mtp:
            raise CapabilityError("Qwen3 text execution disables MTP")
        parent = State(identity, clock, "answer_ready", "decode", _components=components)
        return GemmaDecode(Qwen3Live(self, identity, parent, _owned=True))
