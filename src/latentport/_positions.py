"""Optional sparse-attention layout shared by state operations and backends."""
from .errors import StateError


def physical_length(values, clock):
    """Dense snapshots retain their original ABI; P adds absolute KV positions."""
    if "P" not in values:
        return clock
    import torch
    positions = values["P"]
    if positions.dtype != torch.int64 or positions.ndim != 1 or not 0 < positions.numel() <= clock:
        raise StateError("Invalid retained attention positions")
    if not bool((positions[0] >= 0) & (positions[-1] < clock)):
        raise StateError("Attention positions exceed the logical clock")
    if not bool((positions[1:] > positions[:-1]).all()):
        raise StateError("Attention positions must increase strictly")
    return int(positions.numel())


def continuation_inputs(values, clock, count, *, device, dtype):
    """Use original RoPE coordinates and physical causal-mask columns."""
    import torch
    physical = physical_length(values, clock)
    positions = torch.arange(clock, clock + count, device=device).unsqueeze(0)
    mask = torch.zeros((1, 1, count, physical + count), device=device, dtype=dtype)
    mask[..., physical:] = torch.triu(
        torch.full((count, count), float("-inf"), device=device, dtype=dtype), diagonal=1)
    return {"position_ids": positions,
            "attention_mask": {"full_attention": mask, "linear_attention": None}}


def extend_positions(values, clock, count):
    if "P" not in values:
        return None
    import torch
    return torch.cat((values["P"], torch.arange(clock, clock + count, device=values["P"].device)))
