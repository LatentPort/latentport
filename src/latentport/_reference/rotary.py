"""Static YaRN for standalone decoders with checkpoint-specific rotary layouts."""
from copy import copy

import torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from ..errors import CompatibilityError


def configure_global_yarn(rotary, options, *, layer_type=None):
    if options is None or options.position_policy == "default":
        return

    config = copy(rotary.config)
    if config.max_position_embeddings != options.native_positions:
        raise CompatibilityError("Checkpoint native position range differs from the pinned model")
    prefix = f"{layer_type}_" if layer_type else ""
    native = getattr(rotary, prefix + "inv_freq")
    params = config.rope_parameters[layer_type] if layer_type else config.rope_parameters
    # The existing buffer gives the actual rotary width, including Gemma's
    # unrotated lanes. Flash-Next uses a partial-width buffer for both Q and K.
    config.head_dim = 2 * native.numel()
    config.rope_parameters = dict(rope_type="yarn", rope_theta=params["rope_theta"],
                                  partial_rotary_factor=1.0, factor=options.yarn_factor,
                                  original_max_position_embeddings=options.native_positions)
    frequencies, scale = ROPE_INIT_FUNCTIONS["yarn"](config, native.device)
    if layer_type:
        rotated = native != 0
        frequencies = frequencies.masked_fill(~rotated, 0)
        scale = torch.where(rotated, scale, 1.0).repeat(2)
    setattr(rotary, prefix + "inv_freq", frequencies)
    setattr(rotary, prefix + "original_inv_freq", frequencies.clone())
    setattr(rotary, prefix + "attention_scaling", scale)
