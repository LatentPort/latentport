"""Owned working caches for standalone BF16 and NF4 execution."""
from ..errors import StateError
from .._positions import physical_length, continuation_inputs, extend_positions
from .native import validate_components
from .._models import model_geometry
from . import cache_abi


class LiveCache:
    """Single-owner mutable cache; immutable snapshots are explicit boundaries.

    Installing a State copies it once, preserving forks. Repeated append does
    not clone a whole State, accumulate history IDs or decode historical text.
    A failed mutation poisons this handle; resume from a verified snapshot.
    """
    def __init__(self, backend, identity, state=None, *, _owned=False):
        self.backend, self.identity = backend, identity
        self.geometry = model_geometry(identity.repo_id)
        if state is not None:
            state.require_model(identity)
            backend.options.check(state.clock, state.physical_kv_length)
        self.model = backend._model(identity)
        self.clock = 0
        self.positions = None
        self.terminal = None
        self.parent_id = None
        self.failed = False
        self.cache = self.new_cache(self.model.config)
        if state is not None:
            values = state._read_tensors(device=backend.device) if _owned else state._tensors(device=backend.device)
            validate_components(values, state.clock, "target", geometry=self.geometry, finite=True)
            if state.clock > backend.options.logical_limit or physical_length(values, state.clock) > backend.options.physical_limit:
                raise StateError("Installed Torch state exceeds logical or physical limit")
            self.cache = backend._cache(self.model, values, state.clock)
            self.clock, self.terminal = state.clock, values["H"]
            self.positions, self.parent_id = values.get("P"), state.state_id

    new_cache = staticmethod(cache_abi.new_cache)

    @property
    def physical(self):
        return int(self.cache.get_seq_length())

    def _values(self):
        values = self.backend._extract(self.model, self.cache, self.terminal)
        if self.positions is not None:
            values["P"] = self.positions
        return values

    def append(self, ids):
        # Norm hooks live on the shared model, so their entire lifetime must
        # be serialized, including registration and removal.
        from .native import KERNEL_LOCK
        with KERNEL_LOCK:
            return self._append(ids)

    def _append(self, ids):
        import torch
        ids = tuple(int(x) for x in ids)
        if self.failed or self.cache is None:
            raise StateError("Torch live handle is closed or failed; restore an exact snapshot")
        self.backend.options.check(self.clock, self.physical, len(ids))
        if not ids:
            raise StateError("append requires one or more tokens")
        if len(ids) > 2048:
            raise StateError("Live Torch appends must be at most 2048 tokens")
        terminal = []
        hook = self.model.model.norm.register_forward_pre_hook(
            lambda _, args: terminal.append(args[0][:, -1:].detach().clone()))
        try:
            with torch.inference_mode(), self.backend.kernels():
                # P is not used as mask column indices. All retained keys are
                # earlier logical tokens; only the new chunk is triangular.
                layout = {"P": self.positions} if self.positions is not None else {}
                extra = continuation_inputs(layout, self.clock, len(ids), device=self.backend.device, dtype=torch.bfloat16)
                output = self.model(input_ids=torch.tensor([ids], device=self.backend.device),
                                    past_key_values=self.cache, use_cache=True, logits_to_keep=1, **extra)
                if len(terminal) != 1:
                    raise StateError("Ambiguous Torch terminal capture")
                if self.positions is not None:
                    self.positions = extend_positions({"P": self.positions}, self.clock, len(ids))
                self.cache, self.terminal = output.past_key_values, terminal[0]
                self.clock += len(ids)
                validate_components(self._values(), self.clock, "target", geometry=self.geometry, finite=True)
                return output.logits[:, -1].detach()
        except BaseException:
            self.failed = True
            raise
        finally:
            hook.remove()

    def snapshot(self, *, release=False):
        from ..state import State
        if self.failed or self.terminal is None:
            raise StateError("No valid Torch live state to snapshot")
        values = self._values()
        validate_components(values, self.clock, "target", geometry=self.geometry, finite=True)
        # A released handle transfers exclusive ownership without a second KV
        # allocation. A continuing handle must copy to protect the snapshot.
        if not release:
            values = {k: v.clone() for k, v in values.items()}
        state = State(self.identity, self.clock, "context_ready", self.backend.name+"_live_boundary",
                      target_prompt_tokens=self.clock, parent_id=self.parent_id, _components=values)
        self.parent_id = state.state_id
        if release:
            self.close()
        return state

    def compact(self, *, ratio=8, force=False):
        """Use the public State operation and retain an isolated working cache."""
        parent = self.snapshot(release=True)
        try:
            from .._transaction import transform_install
            child, replacement = transform_install(
                parent, lambda state: state.compact(ratio=ratio, force=force), self.identity,
                lambda state: type(self)(self.backend, self.identity, state))
        except BaseException:
            replacement = type(self)(self.backend, self.identity, parent, _owned=True)
            self.__dict__.update(replacement.__dict__)
            raise
        self.__dict__.update(replacement.__dict__)
        return child

    def close(self):
        self.cache = self.terminal = self.positions = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class LiveDecode:
    def __init__(self, live):
        import torch
        self.live = live
        from .native import KERNEL_LOCK
        with KERNEL_LOCK, torch.inference_mode():
            self.logits = live.model.lm_head(live.model.model.norm(live.terminal))[:, -1]

    @property
    def clock(self):
        return self.live.clock

    def step(self, token_id):
        self.logits = self.live.append([token_id])

    def snapshot(self):
        return self.live.snapshot(release=True)._read_tensors()

    def close(self):
        self.live.close()
