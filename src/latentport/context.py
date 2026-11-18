"""Separate attention capacity from the model's absolute position range."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .errors import CompatibilityError, StateError

NATIVE_POSITIONS = 262144
PINNED_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
YARN4 = {"rope_type": "yarn", "factor": 4.0,
         "original_max_position_embeddings": NATIVE_POSITIONS, "rope_theta": 10000000,
         "partial_rotary_factor": 0.25, "mrope_interleaved": True, "mrope_section": [11, 11, 10]}


@dataclass(frozen=True, slots=True)
class ContextOptions:
    """Static position rules, selected before processing the first token.

    ``physical_limit`` bounds retained KV, not the accumulated token clock.
    Compaction never changes the position policy or resets that clock.
    """
    position_policy: str = "default"
    physical_limit: int = NATIVE_POSITIONS
    max_logical_positions: int | None = None
    native_positions: int = NATIVE_POSITIONS

    def __post_init__(self):
        if type(self.native_positions) is not int or self.native_positions < 1:
            raise ValueError("native_positions must be a positive integer")
        limit = self.max_logical_positions
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("max_logical_positions must be a positive integer")
        if self.position_policy == "yarn":
            if limit is None or limit <= self.native_positions:
                raise ValueError("Position scaling requires a range beyond native positions")
        elif self.position_policy in ("default", "yarn4"):
            ceiling = self.native_positions * (4 if self.position_policy == "yarn4" else 1)
            if limit is not None and limit > ceiling:
                raise ValueError("Requested logical range exceeds the selected position policy")
        else:
            raise ValueError("Unknown position policy")
        if type(self.physical_limit) is not int or not 32 <= self.physical_limit <= NATIVE_POSITIONS:
            raise ValueError("physical_limit must be an integer in 32..262144")

    @property
    def logical_limit(self):
        return self.max_logical_positions or self.native_positions * (4 if self.position_policy == "yarn4" else 1)

    @property
    def legacy(self):
        return (self.position_policy == "default"
                and self.physical_limit == NATIVE_POSITIONS
                and self.logical_limit == self.native_positions)

    def check(self, clock, physical, count=0):
        if any(type(x) is not int for x in (clock, physical, count)) or count < 0 or not 0 <= physical <= clock:
            raise StateError("Invalid continuation coordinates")
        if clock + count > self.logical_limit:
            raise StateError("Logical position limit exceeded")
        if physical + count > self.physical_limit:
            raise StateError("Physical attention limit exceeded; use session.compact(reserve_tokens=...) "
                             "or state.compact(force=True) before appending")

    def record(self):
        rope = self.yarn_parameters() if self.position_policy.startswith("yarn") else json.loads(
            (Path(__file__).parent / "configs/geometry.json").read_text())["target"]["rope"]
        return {"position_policy": self.position_policy, "rope": deepcopy(rope),
                "logical_limit": self.logical_limit, "physical_limit": self.physical_limit,
                "execution": "static-text-positions.v1"}

    @property
    def yarn_factor(self):
        return 4.0 if self.position_policy == "yarn4" else self.logical_limit / self.native_positions

    def yarn_parameters(self):
        return {**deepcopy(YARN4), "factor": self.yarn_factor}


@dataclass(frozen=True, slots=True)
class CheckpointRotaryContext(ContextOptions):
    """Use the pinned checkpoint's global rotary geometry for static YaRN."""

    def record(self):
        return dict(position_policy=self.position_policy, native_positions=self.native_positions,
                    factor=self.yarn_factor, logical_limit=self.logical_limit,
                    physical_limit=self.physical_limit, execution="checkpoint-global-yarn.v1")


def execution_weights(weights, options, *, quantization=None):
    """Keep legacy identities stable; bind new position/quantization semantics."""
    if options.legacy and quantization is None:
        return weights
    record = options.record()
    if quantization is not None:
        record["quantization"] = quantization
    digest = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return weights + ":execution:" + digest


def configure_text(config, options, expected=None):
    """Check exact base geometry, then apply the explicitly selected RoPE rules."""
    from ._reference.architecture import geometry
    from ._reference.native import _geometry
    expected = expected or _geometry("target")
    if geometry(config) != expected or config.max_position_embeddings != NATIVE_POSITIONS:
        raise CompatibilityError("Standalone base geometry or native position range changed")
    config = deepcopy(config)
    if options.position_policy.startswith("yarn"):
        config.rope_parameters = options.yarn_parameters()
        config.max_position_embeddings = options.logical_limit
        expected = {**expected, "rope": config.rope_parameters}
    if geometry(config) != expected:
        raise CompatibilityError("Effective model geometry differs from the selected position policy")
    config.use_cache = True
    config.mtp_num_hidden_layers = 0
    return config
