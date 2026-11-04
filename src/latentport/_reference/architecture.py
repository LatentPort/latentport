"""Preserved research inference definitions; local imports only.

GPU execution of this relocated reference has not yet been requalified.
No model weights are loaded and no CUDA work is started at module import.
"""
from __future__ import annotations
import json
import math
import types
from pathlib import Path
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

HERE = Path(__file__).resolve().parent.parent
def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
def require(condition, message):
    if not condition:
        raise ValueError(message)
GEO = read(HERE / "configs/geometry.json")
CAL = read(HERE / "configs/calibration.json")
CONFIG = read(HERE / "configs/architecture_v2.json")["base_architecture"]
CHUNK = 512
EPS = 1e-6
class PreciseLinear(nn.Linear):
    """BF16 parameter storage with normal connected FP32 computation.

    BF16 shape-dependent GEMM rounding propagated through the coupled map and
    broke the chunk-equivalence gate. Casts remain ordinary autograd edges.
    """

    def forward(self, x):
        return F.linear(x.float(), self.weight.float(), None if self.bias is None else self.bias.float())

@dataclass
class State:
    layers: list[tuple[torch.Tensor, torch.Tensor]]
    terminal: torch.Tensor
    length: int

    def tensors(self):
        return [x for pair in self.layers for x in pair] + [self.terminal]

    def to(self, device, dtype=None):
        return State([tuple((x.to(device=device, dtype=dtype or x.dtype) for x in p)) for p in self.layers], self.terminal.to(device=device, dtype=dtype or self.terminal.dtype), self.length)

def geometry(cfg):
    return {'types': list(cfg.layer_types), 'hidden': cfg.hidden_size, 'kv_heads': cfg.num_key_value_heads, 'head_dim': cfg.head_dim, 'recurrent_heads': cfg.linear_num_value_heads, 'recurrent_k': cfg.linear_key_head_dim, 'recurrent_v': cfg.linear_value_head_dim, 'conv_channels': 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim + cfg.linear_num_value_heads * cfg.linear_value_head_dim, 'conv_width': cfg.linear_conv_kernel_dim, 'rope': cfg.rope_parameters}

class RMSNorm(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-06)).to(x.dtype) * self.weight

class SwiGLU(nn.Module):

    def __init__(self, width, hidden):
        super().__init__()
        self.gate = PreciseLinear(width, hidden, bias=False)
        self.up = PreciseLinear(width, hidden, bias=False)
        self.down = PreciseLinear(hidden, width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Residual(nn.Module):

    def __init__(self, width, hidden):
        super().__init__()
        self.norm = RMSNorm(width)
        self.ffn = SwiGLU(width, hidden)

    def forward(self, x, scale, shift):
        return x + 0.25 * self.ffn(self.norm(x) * (1 + 0.1 * torch.tanh(scale[:, None])) + 0.1 * shift[:, None])

class StateAttention(nn.Module):

    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.q = PreciseLinear(width, width, bias=False)
        self.k = PreciseLinear(width, width, bias=False)
        self.v = PreciseLinear(width, width, bias=False)
        self.o = PreciseLinear(width, width, bias=False)

    def forward(self, query, memory):
        b, n, w = query.shape
        m = memory.shape[1]
        h = self.heads
        q = self.q(query).view(b, n, h, w // h).transpose(1, 2)
        k = self.k(memory).view(b, m, h, w // h).transpose(1, 2)
        v = self.v(memory).view(b, m, h, w // h).transpose(1, 2)
        return self.o(F.scaled_dot_product_attention(q, k, v, dropout_p=0).transpose(1, 2).reshape(b, n, w))

class StateBlock(nn.Module):

    def __init__(self, width, heads, hidden):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.norm2 = RMSNorm(width)
        self.attn = StateAttention(width, heads)
        self.ffn = SwiGLU(width, hidden)

    def forward(self, x):
        y = self.norm1(x)
        x = x + 0.5 * self.attn(y, y)
        return x + 0.5 * self.ffn(self.norm2(x))

class Rope(nn.Module):
    """Exact text-position rotary geometry; compute inverses and rotations in FP32."""

    def __init__(self, geo):
        super().__init__()
        r = geo['rope']
        require(r['rope_type'] == 'default', 'unqualified rotary type')
        self.dim = int(geo['head_dim'] * r.get('partial_rotary_factor', 1.0))
        self.register_buffer('inv_freq', 1 / r['rope_theta'] ** (torch.arange(0, self.dim, 2).float() / self.dim))

    def forward(self, x, inverse=False, start=0):
        t = x.shape[-2]
        angle = torch.arange(start, start + t, device=x.device, dtype=torch.float32)[:, None] * self.inv_freq[None]
        angle = torch.cat((angle, angle), -1)[None, None]
        c = angle.cos().to(x.dtype)
        s = angle.sin().to(x.dtype)
        a = x[..., :self.dim]
        tail = x[..., self.dim:]
        rotate = torch.cat((-a[..., self.dim // 2:], a[..., :self.dim // 2]), -1)
        if inverse:
            y = (a.float() * c.float() - rotate.float() * s.float()) / (c.float().square() + s.float().square())
            return torch.cat((y, tail.float()), -1)
        y = a * c + rotate * s
        return torch.cat((y, tail), -1)

class GCST(nn.Module):

    def __init__(self, source, target, config=None):
        super().__init__()
        c = config or CONFIG
        self.source = source
        self.target = target
        self.skv = [i for i, t in enumerate(source['types']) if t == 'full_attention']
        self.tkv = [i for i, t in enumerate(target['types']) if t == 'full_attention']
        self.slin = [i for i, t in enumerate(source['types']) if t == 'linear_attention']
        self.tlin = [i for i, t in enumerate(target['types']) if t == 'linear_attention']
        w, p, rank = (c['kv_width'], c['persistent_width'], c['tensor_rank'])
        self.chunk = c['context_chunk']
        self.checkpoint_chunks = True
        self.source_rope = Rope(source)
        self.target_rope = Rope(target)
        self.kv_input = PreciseLinear(len(self.skv) * 2 * source['kv_heads'] * source['head_dim'], w, bias=False)
        self.kv_norm = RMSNorm(w)
        self.core = nn.ModuleList((Residual(w, c['kv_ffn']) for _ in range(c['kv_blocks'])))
        self.kv_output = PreciseLinear(w, len(self.tkv) * 2 * target['kv_heads'] * target['head_dim'], bias=False)
        self.rec_input_basis = PreciseLinear(source['recurrent_k'] * source['recurrent_v'], rank, bias=False)
        self.rec_input = PreciseLinear(source['recurrent_heads'] * rank, p, bias=False)
        self.conv_input_basis = PreciseLinear(source['conv_channels'] * source['conv_width'], rank, bias=False)
        self.conv_input = PreciseLinear(rank, p, bias=False)
        self.terminal_input = PreciseLinear(source['hidden'], p, bias=False)
        self.source_identity = nn.Parameter(torch.randn(2 * len(self.slin) + 1, p) * 0.02)
        self.state_encoder = nn.ModuleList((StateBlock(p, c['persistent_heads'], c['persistent_ffn']) for _ in range(c['persistent_blocks'])))
        self.state_norm = RMSNorm(p)
        self.modulation = PreciseLinear(p, 2 * w, bias=False)
        self.final_kv_condition = PreciseLinear(w, p, bias=False)
        self.target_queries = nn.Parameter(torch.randn(2 * len(self.tlin) + 1, p) * 0.02)
        self.query_norm = RMSNorm(p)
        self.memory_norm = RMSNorm(p)
        self.cross = StateAttention(p, c['persistent_heads'])
        self.query_ffn_norm = RMSNorm(p)
        self.query_ffn = SwiGLU(p, c['persistent_ffn'])
        self.decoder_norm = RMSNorm(p)
        self.rec_output = PreciseLinear(p, target['recurrent_heads'] * rank, bias=False)
        self.rec_output_basis = PreciseLinear(rank, target['recurrent_k'] * target['recurrent_v'], bias=False)
        self.conv_output = PreciseLinear(p, rank, bias=False)
        self.conv_output_basis = PreciseLinear(rank, target['conv_channels'] * target['conv_width'], bias=False)
        self.terminal_output = PreciseLinear(p, target['hidden'], bias=False)
        self.register_buffer('source_mean', torch.zeros(len(source['types']), 2, dtype=torch.float32))
        self.register_buffer('source_std', torch.ones(len(source['types']), 2, dtype=torch.float32))
        self.register_buffer('target_mean', torch.zeros(len(target['types']), 2, dtype=torch.float32))
        self.register_buffer('target_std', torch.ones(len(target['types']), 2, dtype=torch.float32))
        self.register_buffer('terminal_stats', torch.tensor([0.0, 1.0, 0.0, 1.0]))
        self.register_buffer('output_gain', torch.ones(len(target['types']), 2, dtype=torch.float32))
        self.register_buffer('terminal_gain', torch.ones((), dtype=torch.float32))

    def normalize(self, x, layer, family):
        return (x.float() - self.source_mean[layer, family]) / self.source_std[layer, family].clamp_min(1e-08)

    def denormalize(self, x, layer, family):
        return (x.float() * self.output_gain[layer, family] * self.target_std[layer, family] + self.target_mean[layer, family]).to(x.dtype)

    def persistent(self, state):
        tokens = []
        for i in self.slin:
            conv, rec = state.layers[i]
            b = rec.shape[0]
            tokens.append(self.conv_input(self.conv_input_basis(self.normalize(conv, i, 0).reshape(b, -1))))
            z = self.rec_input_basis(self.normalize(rec, i, 1).flatten(-2))
            tokens.append(self.rec_input(z.flatten(1)))
        terminal = (state.terminal.float() - self.terminal_stats[0]) / self.terminal_stats[1].clamp_min(1e-08)
        tokens.append(self.terminal_input(terminal[:, -1]))
        x = torch.stack(tokens, 1) + self.source_identity[None]
        for block in self.state_encoder:
            x = block(x)
        x = self.state_norm(x)
        return (x, x.mean(1))

    def kv_chunk(self, flat, scale, shift):
        x = self.kv_input(flat)
        for block in self.core:
            x = block(x, scale, shift)
        x = self.kv_norm(x)
        return (self.kv_output(x), x[:, -1])

    def forward(self, state, chunk=None):
        require(len(state.layers) == len(self.source['types']), 'source family omitted')
        memory, g = self.persistent(state)
        scale, shift = self.modulation(g).chunk(2, -1)
        b = state.terminal.shape[0]
        parts = []
        last = None
        chunk = chunk or self.chunk
        for start in range(0, state.length, chunk):
            end = min(start + chunk, state.length)
            items = []
            for i in self.skv:
                k, v = state.layers[i]
                k = self.source_rope(k[..., start:end, :], inverse=True, start=start)
                items.extend((self.normalize(x, i, j).transpose(1, 2).flatten(2) for j, x in enumerate((k, v[..., start:end, :]))))
            flat = torch.cat(items, -1)
            if self.training and torch.is_grad_enabled() and self.checkpoint_chunks:
                y, last = checkpoint(self.kv_chunk, flat, scale, shift, use_reentrant=False)
            else:
                y, last = self.kv_chunk(flat, scale, shift)
            parts.append(y)
        joined = torch.cat(parts, 1).view(b, state.length, len(self.tkv), 2, self.target['kv_heads'], self.target['head_dim'])
        layers = [None] * len(self.target['types'])
        for j, i in enumerate(self.tkv):
            k = self.denormalize(joined[:, :, j, 0].transpose(1, 2), i, 0).to(self.kv_input.weight.dtype)
            v = self.denormalize(joined[:, :, j, 1].transpose(1, 2), i, 1).to(self.kv_input.weight.dtype)
            layers[i] = (self.target_rope(k), v)
        final = self.final_kv_condition(last)
        memory = memory + final[:, None]
        queries = self.target_queries[None].expand(b, -1, -1) + g[:, None] + final[:, None]
        q = queries + self.cross(self.query_norm(queries), self.memory_norm(memory))
        q = self.decoder_norm(q + 0.5 * self.query_ffn(self.query_ffn_norm(q)))
        for j, i in enumerate(self.tlin):
            conv = self.conv_output_basis(self.conv_output(q[:, 2 * j])).view(b, self.target['conv_channels'], self.target['conv_width'])
            rec = self.rec_output_basis(self.rec_output(q[:, 2 * j + 1]).view(b, self.target['recurrent_heads'], -1))
            rec = rec.view(b, self.target['recurrent_heads'], self.target['recurrent_k'], self.target['recurrent_v'])
            layers[i] = (self.denormalize(conv, i, 0).to(self.kv_input.weight.dtype), self.denormalize(rec, i, 1).float())
        terminal = self.terminal_output(q[:, -1:])
        terminal = (terminal.float() * self.terminal_gain * self.terminal_stats[3] + self.terminal_stats[2]).to(self.kv_input.weight.dtype)
        return State(layers, terminal, state.length)

    def parameter_summary(self):
        groups = {}
        for name, p in self.named_parameters():
            prefix = name.split('.')[0]
            groups[prefix] = groups.get(prefix, 0) + p.numel()
        return {'trainable_parameters': sum((p.numel() for p in self.parameters() if p.requires_grad)), 'groups': groups, 'global_kv_source_layers': self.skv, 'global_kv_target_layers': self.tkv, 'source_persistent_tokens': len(self.source_identity), 'target_state_queries': len(self.target_queries)}

    def parameter_dtype(self, dtype):
        """Keep RoPE and calibration buffers in FP32 while selecting weight dtype."""
        with torch.no_grad():
            for p in self.parameters():
                p.data = p.data.to(dtype=dtype)
        return self

@dataclass(frozen=True)
class SlotMetadata:
    """Numeric raw-input positional dependency, explicitly not a state-only ABI."""
    source_length: int
    target_length: int
    centers: torch.Tensor
    target_positions: torch.Tensor

    def validate(self, source_length=None):
        n, m = (self.source_length, self.target_length)
        if n <= 0 or m <= 0 or (source_length is not None and n != source_length):
            raise ValueError('Invalid native source/target slot count')
        if self.centers.shape != (m,) or self.target_positions.shape != (m,):
            raise ValueError('Metadata must have exactly one center/position per native target slot')
        if not self.centers.is_floating_point() or self.target_positions.dtype != torch.int64:
            raise ValueError('Centers must be floating point; absolute positions must be int64')
        if not bool(torch.isfinite(self.centers).all()) or not bool(((self.centers >= 0) & (self.centers <= n - 1)).all()):
            raise ValueError('Nonfinite or out-of-source-range centers')
        if not bool((self.centers[1:] >= self.centers[:-1]).all()):
            raise ValueError('Alignment prior must be monotonic')
        if not torch.equal(self.target_positions, torch.arange(m, device=self.target_positions.device)):
            raise ValueError('Pinned text prefill ABI requires exact contiguous native target positions starting at zero')
        return self

    def to(self, device):
        return SlotMetadata(self.source_length, self.target_length, self.centers.to(device=device, dtype=torch.float32), self.target_positions.to(device=device))

    @classmethod
    def engineering(cls, n, m, device='cpu'):
        """Shape-only prior for synthetic Gate 0; not a native alignment claim."""
        return cls(n, m, torch.linspace(0, n - 1, m, device=device), torch.arange(m, device=device))

def window_indices(centers, n, radius):
    indices = centers.floor().long()[:, None] + torch.arange(-radius, radius + 1, device=centers.device)[None]
    mask = (indices >= 0) & (indices < n)
    return (indices.clamp(0, n - 1), mask)

def gathered_attention(q, k, v, indices, valid, bias):
    """q B,Q,D; k/v B,N,D; indices Q,W. Work/storage O(Q*W*D)."""
    keys, values = (k[:, indices], v[:, indices])
    scores = (q[:, :, None] * keys).sum(-1) / math.sqrt(q.shape[-1]) + bias[None]
    scores = scores.masked_fill(~valid[None], float('-inf'))
    weights = scores.softmax(-1)
    return ((weights[..., None] * values).sum(-2), weights)

class LocalAttention(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.q = PreciseLinear(width, width, bias=False)
        self.k = PreciseLinear(width, width, bias=False)
        self.v = PreciseLinear(width, width, bias=False)
        self.o = PreciseLinear(width, width, bias=False)

    def forward(self, query, memory, indices, valid, bias):
        value, weights = gathered_attention(self.q(query), self.k(memory), self.v(memory), indices, valid, bias)
        return (self.o(value), weights)

class SparsePositionRouter(nn.Module):

    def __init__(self, content_width, persistent_width, config):
        super().__init__()
        self.config = dict(config)
        r = config['width']
        self.down = PreciseLinear(content_width, r, bias=False)
        self.up = PreciseLinear(r, content_width, bias=False)
        self.global_projection = PreciseLinear(persistent_width, r, bias=False)
        self.position = PreciseLinear(10, r, bias=False)
        self.source_norm = RMSNorm(r)
        self.query_norm = RMSNorm(r)
        self.mixed_norm = RMSNorm(r)
        self.mixer = LocalAttention(r)
        self.route = LocalAttention(r)
        self.offset = PreciseLinear(r, 1, bias=False)
        self.pool_bias = nn.Parameter(torch.full((len(config['pool_scales']) + 1,), -4.0))

    def pool_bank(self, z, g):
        """Exact ragged means; internal pooling padding never creates cache slots."""
        n = z.shape[1]
        parts = [z]
        starts = []
        counts = []
        cursor = n
        for scale in self.config['pool_scales']:
            groups = (n + scale - 1) // scale
            pad = groups * scale - n
            padded = torch.nn.functional.pad(z, (0, 0, 0, pad))
            sums = padded.view(z.shape[0], groups, scale, z.shape[-1]).sum(2)
            sizes = (n - torch.arange(groups, device=z.device) * scale).clamp(max=scale).to(z.dtype)
            parts.append(sums / sizes[None, :, None])
            starts.append(cursor)
            counts.append(groups)
            cursor += groups
        parts.append(self.global_projection(g)[:, None] + z.mean(1, keepdim=True))
        return (torch.cat(parts, 1), starts, cursor)

    def neighborhood(self, centers, n, starts, global_index, radius, corrected=None):
        indices, valid = window_indices(centers, n, radius)
        point = centers if corrected is None else corrected
        bias = -(indices.float() - point[:, None]).square() / (2 * self.config['prior_sigma'] ** 2)
        extras = [centers.floor().long() // scale + start for scale, start in zip(self.config['pool_scales'], starts)]
        extras.append(torch.full_like(indices[:, 0], global_index))
        indices = torch.cat((indices, torch.stack(extras, 1)), 1)
        valid = torch.cat((valid, torch.ones(len(centers), len(extras), device=centers.device, dtype=torch.bool)), 1)
        bias = torch.cat((bias, self.pool_bias[None].expand(len(centers), -1)), 1)
        return (indices, valid, bias)

    def position_features(self, metadata):
        p = metadata.target_positions.float()
        source = metadata.centers / max(1, metadata.source_length - 1)
        target = p / max(1, metadata.target_length - 1)
        return torch.stack((source, target, torch.sin(p), torch.cos(p), torch.sin(p / 16), torch.cos(p / 16), torch.sin(p / 256), torch.cos(p / 256), torch.full_like(p, math.log1p(metadata.source_length) / 16), torch.full_like(p, math.log1p(metadata.target_length) / 16)), -1)

    def forward(self, content, g, metadata, chunk, checkpoint_chunks=True, diagnostics=False):
        metadata.validate(content.shape[1])
        n, m = (metadata.source_length, metadata.target_length)
        z = self.source_norm(self.down(content))
        bank, starts, global_index = self.pool_bank(z, g)
        mix_keys, mix_values = (self.mixer.k(bank), self.mixer.v(bank))
        mixed = []
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            centers = torch.arange(start, end, device=z.device, dtype=torch.float32)
            indices, valid, bias = self.neighborhood(centers, n, starts, global_index, self.config['source_radius'])

            def mix(query, keys, values, b, idx=indices, mask=valid):
                value, _ = gathered_attention(self.mixer.q(query), keys, values, idx, mask, b)
                return self.mixer.o(value)
            value = checkpoint(mix, z[:, start:end], mix_keys, mix_values, bias, use_reentrant=False) if self.training and checkpoint_chunks and torch.is_grad_enabled() else mix(z[:, start:end], mix_keys, mix_values, bias)
            mixed.append(z[:, start:end] + 0.25 * value)
        z = self.mixed_norm(torch.cat(mixed, 1))
        bank, starts, global_index = self.pool_bank(z, g)
        route_keys, route_values = (self.route.k(bank), self.route.v(bank))
        anchor_indices = metadata.centers.round().long()
        anchor = content[:, anchor_indices]
        query = self.query_norm(z[:, anchor_indices] + self.position(self.position_features(metadata))[None] + self.global_projection(g)[:, None])
        offsets = self.config['offset_limit'] * torch.tanh(self.offset(query).squeeze(-1))
        corrected = torch.cummax(metadata.centers[None] + offsets, dim=1).values.clamp(0, n - 1)
        values = []
        off_diagonal = []
        for start in range(0, m, chunk):
            end = min(start + chunk, m)
            centers = metadata.centers[start:end]
            indices, valid, _ = self.neighborhood(centers, n, starts, global_index, self.config['target_radius'])
            local = 2 * self.config['target_radius'] + 1

            def route(q, keys, values, points, idx=indices, mask=valid, _local=local):
                bias = -(idx[:, :_local].float()[None] - points[:, :, None]).square() / (2 * self.config['prior_sigma'] ** 2)
                extra = self.pool_bias[None, None].expand(q.shape[0], q.shape[1], -1)
                bias = torch.cat((bias, extra), -1)
                gathered_keys = keys[:, idx]
                gathered_values = values[:, idx]
                logits = (self.route.q(q)[:, :, None] * gathered_keys).sum(-1) / math.sqrt(q.shape[-1]) + bias
                weights = logits.masked_fill(~mask[None], float('-inf')).softmax(-1)
                return (self.route.o((weights[..., None] * gathered_values).sum(-2)), weights)
            if self.training and checkpoint_chunks and torch.is_grad_enabled() and (not diagnostics):
                routed = checkpoint(lambda q, k, v, p, fn=route: fn(q, k, v, p)[0], query[:, start:end], route_keys, route_values, corrected[:, start:end], use_reentrant=False)
                weights = None
            else:
                routed, weights = route(query[:, start:end], route_keys, route_values, corrected[:, start:end])
            values.append(anchor[:, start:end] + 0.25 * self.up(routed - z[:, anchor_indices[start:end]]))
            if diagnostics:
                diagonal = indices[:, :local] == anchor_indices[start:end, None]
                off_diagonal.append((weights[:, :, :local] * (valid[:, :local] & ~diagonal)[None]).sum(-1).detach())
        result = torch.cat(values, 1)
        details = None
        if diagnostics:
            details = {'off_diagonal_local_mass': torch.cat(off_diagonal, 1), 'learned_offsets': offsets.detach(), 'corrected_centers': corrected.detach(), 'max_local_window': 2 * self.config['target_radius'] + 1, 'pooled_tokens_per_query': len(self.config['pool_scales']) + 1, 'dense_N_by_M_materialized': False}
        return (result, details)

class CrossPositionGCST(GCST):
    """Fresh GCST with sparse positional transformation; architecture has changed."""

    def __init__(self, source, target, base_config, router_config, position_mode='cross_position'):
        super().__init__(source, target, base_config)
        if position_mode not in ('cross_position', 'position_isolated'):
            raise ValueError('Unknown positional ablation')
        self.position_mode = position_mode
        self.position_router = SparsePositionRouter(base_config['kv_width'], base_config['persistent_width'], router_config) if position_mode == 'cross_position' else None

    def encode_chunk(self, flat, scale, shift):
        x = self.kv_input(flat)
        for block in self.core:
            x = block(x, scale, shift)
        return self.kv_norm(x)

    def forward(self, state, slots, chunk=None, diagnostics=False):
        if len(state.layers) != len(self.source['types']):
            raise ValueError('Source layer/family omitted')
        slots = slots.to(state.terminal.device).validate(state.length)
        chunk = int(chunk or self.chunk)
        if chunk <= 0:
            raise ValueError('Chunk must be positive')
        memory, g = self.persistent(state)
        scale, shift = self.modulation(g).chunk(2, -1)
        b = state.terminal.shape[0]
        encoded = []
        for start in range(0, state.length, chunk):
            end = min(start + chunk, state.length)
            items = []
            for i in self.skv:
                k, v = state.layers[i]
                k = self.source_rope(k[..., start:end, :], inverse=True, start=start)
                items.extend((self.normalize(x, i, j).transpose(1, 2).flatten(2) for j, x in enumerate((k, v[..., start:end, :]))))
            flat = torch.cat(items, -1)
            value = checkpoint(self.encode_chunk, flat, scale, shift, use_reentrant=False) if self.training and torch.is_grad_enabled() and self.checkpoint_chunks else self.encode_chunk(flat, scale, shift)
            encoded.append(value)
        content = torch.cat(encoded, 1)
        if self.position_router is None:
            if slots.target_length != state.length or not torch.equal(slots.centers, torch.arange(state.length, device=slots.centers.device, dtype=torch.float32)):
                raise ValueError('Position-isolated A requires equal lengths and identity-compatible slot metadata')
            routed, details = (content, {'position_mode': 'position_isolated', 'off_diagonal_local_mass': torch.zeros(b, state.length, device=content.device)})
        else:
            routed, details = self.position_router(content, g, slots, chunk, self.checkpoint_chunks, diagnostics)
        decoded = []
        for start in range(0, slots.target_length, chunk):
            x = routed[:, start:start + chunk]
            decoded.append(checkpoint(self.kv_output, x, use_reentrant=False) if self.training and torch.is_grad_enabled() and self.checkpoint_chunks else self.kv_output(x))
        joined = torch.cat(decoded, 1).view(b, slots.target_length, len(self.tkv), 2, self.target['kv_heads'], self.target['head_dim'])
        layers = [None] * len(self.target['types'])
        for j, i in enumerate(self.tkv):
            k = self.denormalize(joined[:, :, j, 0].transpose(1, 2), i, 0).to(torch.bfloat16)
            v = self.denormalize(joined[:, :, j, 1].transpose(1, 2), i, 1).to(torch.bfloat16)
            layers[i] = (self.target_rope(k), v)
        final = self.final_kv_condition(routed[:, -1])
        memory = memory + final[:, None]
        queries = self.target_queries[None].expand(b, -1, -1) + g[:, None] + final[:, None]
        q = queries + self.cross(self.query_norm(queries), self.memory_norm(memory))
        q = self.decoder_norm(q + 0.5 * self.query_ffn(self.query_ffn_norm(q)))
        for j, i in enumerate(self.tlin):
            conv = self.conv_output_basis(self.conv_output(q[:, 2 * j])).view(b, self.target['conv_channels'], self.target['conv_width'])
            rec = self.rec_output_basis(self.rec_output(q[:, 2 * j + 1]).view(b, self.target['recurrent_heads'], -1))
            rec = rec.view(b, self.target['recurrent_heads'], self.target['recurrent_k'], self.target['recurrent_v'])
            layers[i] = (self.denormalize(conv, i, 0).to(torch.bfloat16), self.denormalize(rec, i, 1).float())
        terminal = self.terminal_output(q[:, -1:])
        terminal = (terminal.float() * self.terminal_gain * self.terminal_stats[3] + self.terminal_stats[2]).to(torch.bfloat16)
        output = State(layers, terminal, slots.target_length)
        return (output, details) if diagnostics else output

def fresh(calibration, mode='cross_position', seed=890019):
    torch.manual_seed(seed)
    geo = read(HERE / 'configs/geometry.json')
    arch = read(HERE / 'configs/architecture_v2.json')
    with torch.device('cuda'):
        model = CrossPositionGCST(geo['source'], geo['target'], arch['base_architecture'], arch['position_router'], position_mode=mode)
    model.parameter_dtype(torch.float32)
    with torch.no_grad():
        for name, value in calibration['buffers'].items():
            getattr(model, name).copy_(torch.tensor(value, device='cuda', dtype=torch.float32))
    expected = 303623040 if mode == 'position_isolated' else 305756547
    assert model.parameter_summary()['trainable_parameters'] == expected
    return model

class MatrixFactors(torch.nn.Module):
    """Shared nonlinear matrix generator; no global flattened rank-128 basis."""

    def __init__(self, width, k, v, rank):
        super().__init__()
        self.k = k
        self.v = v
        self.rank = rank
        self.left = PreciseLinear(width, k * rank, bias=False)
        self.right = PreciseLinear(width, rank * v, bias=False)

    def forward(self, z):
        a = self.left(z).reshape(*z.shape[:-1], self.k, self.rank)
        b = self.right(z).reshape(*z.shape[:-1], self.rank, self.v)
        return (3 / math.sqrt(self.rank) * (a @ b)).flatten(-2)

class Model(torch.nn.Module):

    def __init__(self, seed, kind, rank, means):
        super().__init__()
        self.base = fresh(CAL, 'cross_position', seed)
        self.kind = kind
        self.means = means
        self.rank = rank
        if kind == 'F':
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(seed + 19000)
                with torch.device('cuda'):
                    self.base.rec_output_basis = MatrixFactors(128, self.base.target['recurrent_k'], self.base.target['recurrent_v'], rank)
        if kind in ('I', 'F'):
            self._identity_hook = self.base.decoder_norm.register_forward_hook(self.layer_identity)

    def layer_identity(self, module, args, q):
        n = 2 * len(self.base.tlin)
        identity = self.base.target_queries[:n]
        identity = identity * torch.rsqrt(identity.square().mean(-1, keepdim=True) + 1e-06)
        mixed = q[:, :n] + identity[None]
        mixed = mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + 1e-06)
        return torch.cat((mixed, q[:, n:]), 1)

    @property
    def target(self):
        return self.base.target

    @property
    def position_mode(self):
        return self.base.position_mode

    def forward(self, source, slots, chunk=128):
        pred = self.base(source, slots, chunk=chunk)
        layers = list(pred.layers)
        for i in self.base.tlin:
            c, r = layers[i]
            layers[i] = ((c.float() - self.base.target_mean[i, 0] + self.means['C.' + str(i)]).to(torch.bfloat16), r - self.base.target_mean[i, 1] + self.means['R.' + str(i)])
        return State(layers, pred.terminal, pred.length)

OriginalF = Model

class HeadBasis(torch.nn.Module):

    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            self.U = torch.nn.Parameter(torch.empty(48, 48, 128, 128, device='cuda', dtype=torch.float32))
            self.V = torch.nn.Parameter(torch.empty_like(self.U))
            torch.nn.init.normal_(self.U, 0.0, 0.1 / math.sqrt(128))
            torch.nn.init.normal_(self.V, 0.0, 0.1 / math.sqrt(128))
            self.g = PreciseLinear(128, 128, bias=True).to(device='cuda', dtype=torch.float32)
        layers = [i for i, t in enumerate(GEO['target']['types']) if t == 'linear_attention']
        scales = [math.sqrt(128 * max(CAL['statistics']['R'][str(i)]['variance'], CAL['statistics']['R'][str(i)]['floor'])) for i in layers]
        self.register_buffer('scale', torch.tensor(scales, device='cuda', dtype=torch.float32))
        self.enabled = True
        self.latest = {}

    def forward(self, z):
        assert z.ndim == 4 and tuple(z.shape[1:]) == (48, 48, 128)
        assert z.device.type == 'cuda' and z.dtype == torch.float32
        coefficients = 1 + 0.1 * torch.tanh(self.g(z))
        self.latest_coefficients = coefficients.detach()
        if not self.enabled:
            return None
        value = self.U[None] * coefficients[..., None, :] @ self.V[None]
        if getattr(self, "collect_diagnostics", False):
            self.latest = dict(g_mean=float(coefficients.detach().mean()), g_std=float(coefficients.detach().std(unbiased=False)), g_min=float(coefficients.detach().min()), g_max=float(coefficients.detach().max()), z_RMS=float(z.detach().square().mean().sqrt()), z_head_L2_min=float(z.detach().norm(dim=-1).min()), z_head_L2_max=float(z.detach().norm(dim=-1).max()))
        return value * self.scale[None, :, None, None, None]

class HeadBasisModel(OriginalF):

    def __init__(self, seed, means, new_seed):
        super().__init__(seed, 'F', 96, means)
        self.head_basis = HeadBasis(new_seed)
        self._connected_latents = None
        self._capture = self.base.rec_output_basis.register_forward_pre_hook(self._latent_hook)

    def _latent_hook(self, module, args):
        if self._connected_latents is not None:
            self._connected_latents.append(args[0])

    def forward(self, source, slots, chunk=128):
        assert self._connected_latents is None, 'Nested forwards are not supported'
        self._connected_latents = []
        try:
            pred = super().forward(source, slots, chunk)
            assert len(self._connected_latents) == 48
            z = torch.stack(self._connected_latents, dim=1)
            delta = self.head_basis(z)
        finally:
            self._connected_latents = None
        if delta is None:
            return pred
        layers = list(pred.layers)
        for j, i in enumerate(self.base.tlin):
            c, r = layers[i]
            assert r.dtype == torch.float32 and r.shape == delta[:, j].shape
            layers[i] = (c, r + delta[:, j])
        return State(layers, pred.terminal, pred.length)

    def recurrent_only(self):
        for name, p in self.named_parameters():
            p.requires_grad_(name.startswith(('head_basis.', 'base.rec_output.', 'base.rec_output_basis.')))

def historical_source(source):
    import torch
    return State([tuple((x.to(torch.bfloat16) for x in pair)) for pair in source.layers], source.terminal.to(torch.bfloat16), source.length)

def base_forward(self, state, slots, chunk=None, diagnostics=False):
    import torch
    if self.training or torch.is_grad_enabled():
        raise RuntimeError('Inference lifetime scheduling only')
    if len(state.layers) != len(self.source['types']):
        raise ValueError('Source layer/family omitted')
    slots = slots.to(state.terminal.device).validate(state.length)
    chunk = int(chunk or CHUNK)
    if chunk != CHUNK:
        raise ValueError('Pinned 512-token translator scheduling required')
    memory, g = self.persistent(state)
    scale, shift = self.modulation(g).chunk(2, -1)
    b = state.terminal.shape[0]
    encoded = []
    for start in range(0, state.length, chunk):
        end = min(start + chunk, state.length)
        items = []
        for i in self.skv:
            k, v = state.layers[i]
            k = self.source_rope(k[..., start:end, :], inverse=True, start=start)
            items.extend((self.normalize(x, i, j).transpose(1, 2).flatten(2) for j, x in enumerate((k, v[..., start:end, :]))))
        flat = torch.cat(items, -1)
        encoded.append(self.encode_chunk(flat, scale, shift))
        del items, flat, k, v
    content = torch.cat(encoded, 1)
    del encoded
    if getattr(self, '_e096_consume_source', False):
        state.layers.clear()
        state.terminal = None
    if self.position_router is None:
        raise RuntimeError('Frozen cross-position router required')
    routed, details = self.position_router(content, g, slots, chunk, self.checkpoint_chunks, diagnostics)
    del content
    layers = [None] * len(self.target['types'])
    if getattr(self, '_emit_kv', True):
        pieces = {i: [[], []] for i in self.tkv}
        for start in range(0, slots.target_length, chunk):
            end = min(start + chunk, slots.target_length)
            joined = self.kv_output(routed[:, start:end]).view(b, end - start, len(self.tkv), 2, self.target['kv_heads'], self.target['head_dim'])
            for j, i in enumerate(self.tkv):
                k = self.denormalize(joined[:, :, j, 0].transpose(1, 2), i, 0).to(torch.bfloat16)
                v = self.denormalize(joined[:, :, j, 1].transpose(1, 2), i, 1).to(torch.bfloat16)
                pieces[i][0].append(self.target_rope(k, start=start))
                pieces[i][1].append(v)
            del joined, k, v
        for i in self.tkv:
            pair = pieces.pop(i)
            layers[i] = (torch.cat(pair[0], -2), torch.cat(pair[1], -2))
            del pair
    final = self.final_kv_condition(routed[:, -1])
    del routed
    memory = memory + final[:, None]
    queries = self.target_queries[None].expand(b, -1, -1) + g[:, None] + final[:, None]
    q = queries + self.cross(self.query_norm(queries), self.memory_norm(memory))
    q = self.decoder_norm(q + 0.5 * self.query_ffn(self.query_ffn_norm(q)))
    for j, i in enumerate(self.tlin):
        conv = self.conv_output_basis(self.conv_output(q[:, 2 * j])).view(b, self.target['conv_channels'], self.target['conv_width'])
        rec = self.rec_output_basis(self.rec_output(q[:, 2 * j + 1]).view(b, self.target['recurrent_heads'], -1))
        rec = rec.view(b, self.target['recurrent_heads'], self.target['recurrent_k'], self.target['recurrent_v'])
        layers[i] = (self.denormalize(conv, i, 0).to(torch.bfloat16), self.denormalize(rec, i, 1).float())
    terminal = self.terminal_output(q[:, -1:])
    terminal = (terminal.float() * self.terminal_gain * self.terminal_stats[3] + self.terminal_stats[2]).to(torch.bfloat16)
    result = State(layers, terminal, slots.target_length)
    return (result, details) if diagnostics else result

def install(translator, *, consume_source=False):
    if not hasattr(translator.base, '_e096_original_forward'):
        translator.base._e096_original_forward = translator.base.forward
    translator.base.forward = types.MethodType(base_forward, translator.base)
    translator.base._e096_consume_source = consume_source
    return translator

class Head(torch.nn.Module):

    def __init__(self, geometry, values, device='cuda'):
        super().__init__()
        self.target = geometry
        self.tkv = [i for i, v in enumerate(geometry['types']) if v == 'full_attention']
        weight = values['base.kv_output.weight']
        self.kv_output = PreciseLinear(weight.shape[1], weight.shape[0], bias=False, device=device, dtype=torch.float32)
        with torch.no_grad():
            self.kv_output.weight.copy_(weight.to(device))
        for name in ('output_gain', 'target_std', 'target_mean'):
            self.register_buffer(name, values['base.' + name].to(device=device, dtype=torch.float32, copy=True))
        self.target_rope = Rope(geometry).to(device)
        with torch.no_grad():
            self.target_rope.inv_freq.copy_(values['base.target_rope.inv_freq'].to(device))

    def forward(self, features, start):
        batch, length = features.shape[:2]
        joined = self.kv_output(features).view(batch, length, len(self.tkv), 2, self.target['kv_heads'], self.target['head_dim'])
        result = {}
        for j, layer in enumerate(self.tkv):
            k = GCST.denormalize(self, joined[:, :, j, 0].transpose(1, 2), layer, 0).to(torch.bfloat16)
            v = GCST.denormalize(self, joined[:, :, j, 1].transpose(1, 2), layer, 1).to(torch.bfloat16)
            result[f'K.{layer}'] = self.target_rope(k, start=start)
            result[f'V.{layer}'] = v
        return result

def slots_for(metadata, device):
    import torch
    return SlotMetadata(metadata['source_length'], metadata['target_length'], torch.tensor(metadata['centers'], device=device, dtype=torch.float32), torch.tensor(metadata['target_positions'], device=device, dtype=torch.int64)).validate(metadata['source_length'])

def routed_features(model, source, metadata):
    """Original encoder/router equations. No H/C/R output call or native content.

    Restore R as FP32; the frozen historical translator ingress then casts it
    to BF16, exactly as in E096. Only existing K/V modules receive gradients.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    base = model.base
    state = historical_source(source)
    slots = slots_for(metadata, state.terminal.device)
    with torch.no_grad():
        memory, g = base.persistent(state)
        scale, shift = base.modulation(g).chunk(2, -1)
    del memory
    encoded = []
    for start in range(0, state.length, CHUNK):
        end = min(start + CHUNK, state.length)
        with torch.no_grad():
            items = []
            for i in base.skv:
                k, v = state.layers[i]
                k = base.source_rope(k[..., start:end, :], inverse=True, start=start)
                items.extend((base.normalize(x, i, j).transpose(1, 2).flatten(2) for j, x in enumerate((k, v[..., start:end, :]))))
            flat = torch.cat(items, -1)
        if base.training and torch.is_grad_enabled() and base.checkpoint_chunks:
            value = checkpoint(base.encode_chunk, flat, scale, shift, use_reentrant=False)
        else:
            value = base.encode_chunk(flat, scale, shift)
        encoded.append(value)
    content = torch.cat(encoded, 1)
    routed, _ = base.position_router(content, g, slots, CHUNK, base.checkpoint_chunks, False)
    return routed

class RMSLinearHResidual(nn.Module):

    def __init__(self, width=5120, rank=16):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False, dtype=torch.float32)
        self.up = nn.Linear(rank, width, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.up.weight)

    def rank_features(self, h):
        h = h.float()
        normalized = h * torch.rsqrt(h.square().mean(dim=-1, keepdim=True) + EPS)
        return self.down(normalized)

    def forward(self, source_derived_H):
        if source_derived_H.shape[-1] != self.down.in_features:
            raise ValueError('H_WIDTH')
        h = source_derived_H.float()
        return (h + self.up(self.rank_features(h))).to(source_derived_H.dtype)
