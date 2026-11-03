"""Public LatentPort exceptions."""


class LatentPortError(RuntimeError):
    """Base class for an operation that cannot satisfy its stated contract."""


class CompatibilityError(LatentPortError):
    """Model, tokenizer, bridge, quantization or state ABI does not match."""


class StateError(LatentPortError):
    """A state is malformed, at the wrong boundary, or cannot be restored."""


class CapabilityError(LatentPortError):
    """The selected runtime does not implement an explicitly requested operation."""


class ReplayError(LatentPortError):
    """A backend tried to prefill historical tokens through a receiving model."""


class HandoffError(LatentPortError):
    """Translation or receiver installation failed; the original session is intact."""
