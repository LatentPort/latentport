"""Public LatentPort exceptions."""


class LatentPortError(RuntimeError):
    """Base class for an operation that cannot satisfy its stated contract."""


class CompatibilityError(LatentPortError):
    """Model, tokenizer, bridge, quantization or state ABI does not match."""


class StateError(LatentPortError):
    """A state is malformed, at the wrong boundary, or cannot be restored."""


class GenerationInterrupted(StateError):
    """Stream output was exposed, but the request rolled back.

    ``token_ids`` includes every callback invocation, including one that raised.
    Discard that request's output before retrying from ``recovery_state``. The
    original failure is available as ``__cause__``; delivery is not acknowledged
    merely because a callback returned.
    """

    def __init__(self, token_ids, recovery_state):
        super().__init__("Generation interrupted; discard this request's streamed output before retrying")
        self.token_ids = tuple(token_ids)
        self.recovery_state = recovery_state


class CapabilityError(LatentPortError):
    """The selected runtime does not implement an explicitly requested operation."""


class ReplayError(LatentPortError):
    """A backend tried to prefill historical tokens through a receiving model."""


class HandoffError(LatentPortError):
    """Translation or receiver installation failed; the original session is intact."""
