"""Layer-aligned 4B→9B writer in unrotated key coordinates."""
import torch
from torch import nn
from .architecture import Rope, State


class Writer(nn.Module):
    def __init__(self, source, target):
        super().__init__()
        for key in ("types", "kv_heads", "head_dim", "recurrent_heads", "recurrent_k",
                    "recurrent_v", "conv_channels", "conv_width"):
            if source[key] != target[key]:
                raise ValueError("Layer-aligned writer requires matching " + key)
        self.types = target["types"]
        self.source_rope, self.target_rope = Rope(source), Rope(target)
        n = self.types.count("full_attention")
        m = len(self.types) - n
        d, h = target["head_dim"], target["kv_heads"]
        self.kv = nn.Parameter(torch.eye(d).repeat(n, 2, h, 1, 1))
        self.bias = nn.Parameter(torch.zeros(n, 2, h, 1, d))
        self.h_projection = nn.Linear(source["hidden"], target["hidden"])
        width = target["hidden"]
        self.h_down = nn.Parameter(torch.randn(2, 16, width) / width**0.5)
        self.h_up = nn.Parameter(torch.zeros(2, width, 16))
        conv_shape = (m, target["conv_channels"], target["conv_width"])
        self.register_buffer("conv_base", torch.ones(conv_shape))
        self.register_buffer("conv_offset", torch.zeros(conv_shape))
        self.register_buffer("rec_scale", torch.ones(m, target["recurrent_heads"], target["recurrent_k"], 1))
        self.conv_gain = nn.Parameter(torch.zeros(2, *conv_shape))
        self.conv_bias = nn.Parameter(torch.zeros(2, *conv_shape))
        rk, rv, heads = target["recurrent_k"], target["recurrent_v"], target["recurrent_heads"]
        self.left_down = nn.Parameter(torch.randn(2, m, heads, 4, rk) / rk**0.5)
        self.left_up = nn.Parameter(torch.zeros(2, m, heads, rk, 4))
        self.right_down = nn.Parameter(torch.randn(2, m, heads, rv, 4) / rv**0.5)
        self.right_up = nn.Parameter(torch.zeros(2, m, heads, 4, rv))

    def hidden(self, terminal, mode):
        h = self.h_projection(terminal.float())
        unit = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-6)
        return (h + (unit @ self.h_down[mode].T) @ self.h_up[mode].T).to(terminal.dtype)

    def forward(self, state, mode, direct=False):
        pairs = []
        att = rec = 0
        for layer, kind in enumerate(self.types):
            a, b = state.layers[layer]
            if kind == "full_attention":
                k = self.source_rope(a, inverse=True) @ self.kv[att, 0] + self.bias[att, 0]
                v = b.float() @ self.kv[att, 1] + self.bias[att, 1]
                pairs.append((self.target_rope(k.to(a.dtype)), v.to(b.dtype)))
                att += 1
            else:
                c, r = a, b
                if not direct:
                    c = (a.float() * self.conv_base[rec] + self.conv_offset[rec])
                    c = c * (1 + self.conv_gain[mode, rec]) + self.conv_bias[mode, rec]
                    r = b.float() * self.rec_scale[rec]
                    r = r + self.left_up[mode, rec] @ (self.left_down[mode, rec] @ r) + (
                        r @ self.right_down[mode, rec]) @ self.right_up[mode, rec]
                pairs.append((c.to(a.dtype), r.to(b.dtype)))
                rec += 1
        return State(pairs, self.hidden(state.terminal, mode), state.length)





class LayerAlignedTranslator:
    def __init__(self, root):
        import json
        self.root = root
        self.config = json.loads((root / "bridge_config.json").read_text())
        self.geo = json.loads((root / "configs/geometry.json").read_text())
        self.writer = None

    def warm(self):
        from safetensors.torch import load_file
        if self.writer is None:
            writer = Writer(self.geo["source"], self.geo["target"])
            writer.load_state_dict(load_file(self.root / self.config["weights"]["writer"]), strict=True)
            self.writer = writer.cuda().eval().requires_grad_(False)

    @torch.inference_mode()
    def translate(self, components, source_clock, metadata):
        from .native import validate_components
        self.warm()
        validate_components(components, source_clock, "source", geometry=self.geo["source"])
        fields = [("K", "V") if kind == "full_attention" else ("C", "R")
                  for kind in self.geo["source"]["types"]]
        layers = [tuple(components[f"{k}.{i}"] for k in pair) for i, pair in enumerate(fields)]
        mode = self.config["thinking_modes"].index(metadata["thinking_mode"])
        target = self.writer(State(layers, components["H"], source_clock), mode)
        values = {"H": target.terminal}
        for i, pair in enumerate(fields):
            for key, tensor in zip(pair, target.layers[i], strict=True):
                values[f"{key}.{i}"] = tensor
        validate_components(values, source_clock, "target", finite=True, geometry=self.geo["target"])
        return values
