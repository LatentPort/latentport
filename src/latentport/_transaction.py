"""Shared immutable-state transformation and installation boundary."""
from .errors import StateError
from .state import State


def transform_install(parent, operation, identity, install):
    """Return a validated state and prepared handle; caller commits atomically.

    Operations must preserve borrowed input tensors. The parent remains owned
    by the caller on any exception, including destination allocation failure.
    Installers must release partial resources when raising.
    """
    child = operation(parent)
    if not isinstance(child, State):
        raise StateError("State transformation did not return a State")
    child.require_model(identity)
    handle = install(child)
    return child, handle
