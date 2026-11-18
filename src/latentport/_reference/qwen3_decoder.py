"""Native-position BF16 Qwen3 decoder and immutable cache branches."""

from contextlib import nullcontext

import torch
from transformers import Qwen3ForCausalLM

from .rotary import configure_global_yarn


class Decoder:
    options = None

    def __init__(self, snapshot, device="cuda:0", *, options=None, nf4=False):
        self.options = options
        self.device = torch.device(device)
        backend = {}
        if nf4:
            from .nf4 import transformers_config
            backend["quantization_config"] = transformers_config()
        self.model = (
            Qwen3ForCausalLM.from_pretrained(
                snapshot,
                dtype=torch.bfloat16,
                device_map={"": str(self.device)},
                attn_implementation="sdpa",
                local_files_only=True,
                **backend,
            )
            .eval()
            .requires_grad_(False)
        )
        self.text = self.model.model
        self.cfg = self.config = self.model.config
        configure_global_yarn(self.text.rotary_emb, options)

    def layer(self, idx):
        return nullcontext(self.text.layers[idx])

    @torch.inference_mode()
    def step(self, ids, cache, clock, *, keep_last=1):
        if clock + len(ids) > getattr(self.options, "logical_limit", self.cfg.max_position_embeddings):
            raise ValueError("Native Qwen3 position limit exceeded")
        tokens = torch.as_tensor(ids, device=self.device).reshape(1, -1)
        positions = torch.arange(clock, clock + len(ids), device=self.device)[None]
        hidden = self.text.embed_tokens(tokens)
        cos, sin = self.text.rotary_emb(hidden, positions)
        for i, layer in enumerate(self.text.layers):
            slot = cache.layers[i]
            for start in range(0, len(ids), 512):
                end = min(start + 512, len(ids))
                prior = slot.keys.shape[-2] if slot.is_initialized else 0
                cols = torch.arange(prior + end - start, device=self.device)[None]
                rows = prior + torch.arange(end - start, device=self.device)[:, None]
                hidden[:, start:end] = layer(
                    hidden[:, start:end],
                    position_embeddings=(cos[:, start:end], sin[:, start:end]),
                    attention_mask=(cols <= rows)[None, None],
                    position_ids=positions[:, start:end],
                    past_key_values=cache,
                )
        return self.text.norm(hidden[:, -keep_last:])

    def logits(self, hidden):
        return self.model.lm_head(hidden)
