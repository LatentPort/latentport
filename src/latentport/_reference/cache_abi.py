"""Complete Qwen3.5 cache reconstruction, including linear-attention state."""
from __future__ import annotations

import functools
import weakref


def _update_conv(layer, x, state_idx=0, conv_kernel_size=None, **kwargs):
    import torch.nn.functional as F
    width = int(conv_kernel_size or layer.conv_kernel_size[state_idx] or x.shape[-1])
    if not layer.is_conv_states_initialized[state_idx]:
        layer.is_conv_states_initialized[state_idx] = True
        layer.conv_kernel_size[state_idx] = width
    if layer.has_previous_state[state_idx]:
        full = __import__("torch").cat((layer.conv_states[state_idx], x), -1)
    else:
        full = x
        layer.has_previous_state[state_idx] = True
    padded = F.pad(full, (max(0, width - full.shape[-1]), 0))
    layer.conv_states[state_idx] = padded[..., -width:]
    layer.dtype, layer.device = x.dtype, x.device
    return full


def _update_recurrent(layer, x, state_idx=0, **kwargs):
    layer.recurrent_states[state_idx] = x
    layer.is_recurrent_states_initialized[state_idx] = True
    layer.dtype, layer.device = x.dtype, x.device
    return x


def new_cache(config):
    from transformers import DynamicCache
    from transformers.cache_utils import LinearAttentionCacheLayerMixin
    result = DynamicCache(config=config, offloading=False)
    for layer in result.layers:
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            proxy = weakref.proxy(layer)
            layer.update_conv_state = functools.partial(_update_conv, proxy)
            layer.update_recurrent_state = functools.partial(_update_recurrent, proxy)
            layer.record_past = True
    return result


def assign(cache, index, kind, a, b):
    layer = cache.layers[index]
    if kind == "full_attention":
        layer.keys, layer.values, layer.is_initialized = a, b, True
    else:
        layer.conv_states[0], layer.recurrent_states[0] = a, b
        layer.is_conv_states_initialized[0] = True
        layer.is_recurrent_states_initialized[0] = True
        layer.has_previous_state[0] = True
        layer.conv_kernel_size[0] = a.shape[-1]
    layer.dtype, layer.device = a.dtype, a.device


def extract(cache, index, kind):
    layer = cache.layers[index]
    return (layer.keys, layer.values) if kind == "full_attention" else (
        layer.conv_states[0], layer.recurrent_states[0])
