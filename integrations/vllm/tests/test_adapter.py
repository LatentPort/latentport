from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from latentport._compact import policy, weights
from latentport.flash_next import _blocks
from latentport.selector import KVSelector
from latentport_vllm.cli import build_args
from latentport_vllm.engine import EngineCompaction
from latentport_vllm.ple import readonly_ple_shard
from latentport_vllm.settings import FLASH_NEXT, FLASH_NEXT_NVFP4, Settings
from latentport_vllm.worker import (
    CompactedRequest,
    PageRows,
    WorkerCompaction,
    _cache_layout,
    _validate_keep,
)


@pytest.mark.parametrize("name", ["27B", "Flash-Next"])
@pytest.mark.parametrize("ratio", [4, 8])
def test_external_selector_matches_native_policy(monkeypatch, name, ratio):
    base = f"Qwen/Qwen3.8-{name}"
    artifact = f"latentport/LPCompact-Qwen3.8-{name}"
    spec = Settings.load(base, {"artifact": artifact}).selector
    generator = torch.Generator().manual_seed(17)
    # Cross a scoring chunk boundary and leave a partial QSA group.
    rows = 8197
    features = [
        torch.randn((rows, 1, spec.feature_width // 3), generator=generator)
        for _ in spec.feature_layers
    ]
    scorer = torch.nn.Linear(spec.feature_width, 1).eval().requires_grad_(False)
    monkeypatch.setattr(weights, "_model", lambda *_: scorer)
    native_values = {
        f"V.{layer}": data.transpose(0, 1).unsqueeze(0)
        for layer, data in zip(spec.feature_layers, features)
    }
    internal_scores = weights.score(native_values, base)
    native_target = policy.budget(rows, ratio)
    expected = (
        _blocks(
            torch.tensor(internal_scores), max(native_target, 1024 + rows % 4)
        ).tolist()
        if name == "Flash-Next"
        else policy.choose(internal_scores, native_target)
    )
    result = KVSelector(spec, scorer).select(
        lambda start, stop: [data[start:stop] for data in features],
        rows,
        ratio,
    )
    assert result == expected
    _validate_keep(result, rows, spec.target_rows(rows, ratio), spec.granularity)


def test_page_rows_respect_noncontiguous_block_ids_and_indexer_units():
    cache = torch.arange(6 * 4).reshape(6, 4, 1, 1)
    compressed = PageRows(0, cache, stride=4)
    old = torch.tensor([3, 0, 4])
    new = torch.tensor([5, 2])
    groups = torch.tensor([1, 2, 7, 8, 10])
    expected = compressed.read(old, groups)
    compressed.write(new, torch.arange(groups.numel()), expected)
    assert compressed.read(new, torch.arange(groups.numel())).flatten().tolist() == [
        13,
        14,
        3,
        16,
        18,
    ]
    assert torch.equal(compressed.read(old, groups), expected)


def test_flash_keep_validation_rejects_a_split_qsa_group():
    with pytest.raises(ValueError, match="split"):
        _validate_keep([0, 1, 2, 4, 12], 13, 5, 4)


def test_tensor_parallel_features_use_each_global_head_once():
    compactor = WorkerCompaction.__new__(WorkerCompaction)
    compactor.kv_heads = 2
    compactor.tp = SimpleNamespace(
        world_size=4,
        all_gather=lambda _, dim: torch.tensor([[[1], [1], [3], [3]]]),
    )
    cache = PageRows(0, torch.zeros(1, 2, 1, 2))
    result = compactor._global_features(cache, torch.tensor([0]), torch.tensor([1]))
    assert result.flatten().tolist() == [1, 3]


def test_packed_metadata_does_not_shift_logical_or_recurrent_coordinates():
    compactor = WorkerCompaction.__new__(WorkerCompaction)
    compactor.device = torch.device("cpu")
    compactor.requests = {"packed": CompactedRequest(10, torch.tensor([0, 8, 9]))}
    compactor.movable_groups = (0, 2)
    compactor.packed_groups = (0, 2, 3)
    compactor.runner = SimpleNamespace(
        block_tables=SimpleNamespace(
            block_sizes=[4, 4, 4, 4],
            _slot_mapping_enabled=[True, True, True, False],
        )
    )
    batch = SimpleNamespace(
        req_ids=["packed", "dense"],
        num_reqs=2,
        num_reqs_after_padding=2,
        num_tokens=3,
        seq_lens=torch.tensor([12, 3]),
        seq_lens_cpu_upper_bound=torch.tensor([12, 3]),
        query_start_loc=torch.tensor([0, 2, 3]),
        positions=torch.tensor([10, 11, 2]),
    )
    tables = (torch.tensor([[4, 8], [2, 0]]),) * 4
    slots = torch.full((4, 3), -1, dtype=torch.long)
    compactor.prepare_attention(batch, tables, slots)
    assert slots.tolist() == [[19, 32, 10], [-1, -1, -1], [19, 32, 10], [-1, -1, -1]]
    assert set(batch.packed_attention_lengths) == {0, 2, 3}
    for lens, upper, maximum in batch.packed_attention_lengths.values():
        assert lens.tolist() == [5, 3]
        assert upper.tolist() == [5, 3]
        assert maximum == 5
    assert batch.positions.tolist() == [10, 11, 2]
    assert batch.seq_lens.tolist() == [12, 3]


def test_launcher_uses_the_selected_artifacts_base_revision():
    argv = build_args(
        [
            "serve",
            "Qwen/Qwen3.8-Flash-Next",
            "--lpcompact",
            "latentport/LPCompact-Qwen3.8-Flash-Next",
            "--tensor-parallel-size",
            "4",
        ]
    )
    assert (
        argv[argv.index("--revision") + 1] == "812d96e10a0dd82b69d749e68693698cf6c36daa"
    )
    assert argv[-2:] == ["--tensor-parallel-size", "4"]
    with pytest.raises(SystemExit):
        build_args(
            [
                "serve",
                "Qwen/Qwen3.8-Flash-Next",
                "--lpcompact",
                "latentport/LPCompact-Qwen3.8-27B",
            ]
        )


def test_nvfp4_launch_keeps_the_flash_next_selector_and_pins_the_served_weights():
    artifact = "latentport/LPCompact-Qwen3.8-Flash-Next"
    settings = Settings.load(FLASH_NEXT_NVFP4, {"artifact": artifact})
    argv = build_args(["serve", FLASH_NEXT_NVFP4, "--lpcompact", artifact])
    assert settings.selector.base_model == FLASH_NEXT
    assert settings.selector.artifact == artifact
    assert (
        argv[argv.index("--revision") + 1] == "103a7608316173ca6edd49929544244de7ffda70"
    )
    assert argv[argv.index("--engram-config") + 1] == '{"disk_offload":true}'


@pytest.mark.parametrize(
    ("model", "artifact", "minimum", "native_positions", "layers", "heads"),
    [
        ("Qwen/Qwen3.5-4B", "Qwen3.5-4B", 65536, 262144, 8, 4),
        ("Qwen/Qwen3.5-9B", "Qwen3.5-9B", 65536, 262144, 8, 4),
        ("Qwen/Qwen3-14B", "Qwen3-14B", 32768, 40960, 40, 8),
        ("Qwen/Qwen3.6-35B-A3B", "Qwen3.6-35B-A3B", 65536, 262144, 10, 2),
        ("google/gemma-4-31B-it", "Gemma-4-31B-it", 65536, 262144, 10, 4),
    ],
)
def test_catalog_profiles_use_their_native_geometry(
    model,
    artifact,
    minimum,
    native_positions,
    layers,
    heads,
):
    artifact = f"latentport/LPCompact-{artifact}"
    settings = Settings.load(model, {"artifact": artifact})
    assert settings.minimum_rows == minimum
    assert settings.selector.native_positions == native_positions
    assert len(settings.selector.attention_layers) == layers
    assert settings.selector.attention_kv_heads == heads
    args = build_args(["serve", model, "--lpcompact", artifact])
    assert args[args.index("--max-model-len") + 1] == str(min(native_positions, 139264))
    assert args[args.index("--revision") + 1] == settings.selector.base_revision


def test_gemma_compaction_selects_only_global_attention_groups():
    settings = Settings.load(
        "google/gemma-4-31B-it",
        {"artifact": "latentport/LPCompact-Gemma-4-31B-it"},
    )
    full = set(settings.selector.attention_layers)
    grouped_names = [[], []]
    layers = {}
    for index in range(60):
        name = f"model.layers.{index}.self_attn.attn"
        group = 0 if index in full else 1
        grouped_names[group].append(name)
        head_size, heads = (512, 4) if group == 0 else (256, 16)
        layers[name] = SimpleNamespace(
            get_attn_backend=lambda: SimpleNamespace(get_name=lambda: "TRITON_ATTN"),
            head_size=head_size,
            kv_cache=torch.empty((1, heads, 4, 2 * head_size), dtype=torch.bfloat16),
        )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context=layers),
        ),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(layer_names=names) for names in grouped_names
            ],
        ),
        block_tables=SimpleNamespace(block_sizes=[4, 4]),
    )
    views, features, moved, packed = _cache_layout(runner, settings)
    assert {view.layer for view in views} == full
    assert tuple(view.layer for view in features) == (5, 29, 59)
    assert moved == packed == (0,)


def test_admission_reserves_physical_sliding_window_capacity():
    settings = Settings.load(
        "google/gemma-4-31B-it",
        {"artifact": "latentport/LPCompact-Gemma-4-31B-it"},
    )
    model_len = 65536
    groups = [
        SimpleNamespace(
            kv_cache_spec=SimpleNamespace(
                max_memory_usage_bytes=lambda _, count=pages: count * 16,
                page_size_bytes=16,
            ),
        )
        for pages in (8, 2)
    ]
    cache = SimpleNamespace(
        block_pool=SimpleNamespace(get_num_free_blocks=lambda: 11),
        coordinator=SimpleNamespace(
            single_type_managers=[SimpleNamespace(block_size=8192)]
        ),
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
    )
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(kv_cache_manager=cache),
        collective_rpc=lambda *args, **kwargs: [(0,)],
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=model_len),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
    )
    assert EngineCompaction(engine, config, settings).groups == (0,)


def test_readonly_ple_shard_preserves_values_and_defers_mixed_files(tmp_path):
    path = tmp_path / "embedding.safetensors"
    name = "model.layers.1.ple.ngram_embedding.shard_0.weight"
    expected = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    save_file({name: expected}, path)
    pairs = readonly_ple_shard(path)
    assert len(pairs) == 1
    assert pairs[0][0] == name
    assert torch.equal(pairs[0][1], expected)
    del pairs
    save_file({name: expected, "attn": expected.clone()}, path)
    assert readonly_ple_shard(path) is None


@pytest.mark.parametrize(
    "outcome", ["commit", "failed copy", "no pages", "same page count"]
)
def test_transaction_releases_only_unowned_pages(outcome):
    source = [SimpleNamespace(block_id=i) for i in (1, 2, 3, 4)]
    reserved = [SimpleNamespace(block_id=10)]
    live = SimpleNamespace(
        block_size=64, req_to_blocks={"r": source}, packed_rows_dropped={}
    )
    frees = []
    events = []

    def rpc(method, *, args):
        events.append((method, args))
        return ["failed" if outcome == "failed copy" else None]

    transaction = EngineCompaction.__new__(EngineCompaction)
    transaction.engine = SimpleNamespace(collective_rpc=rpc)
    transaction.settings = Settings.load(
        "Qwen/Qwen3.8-27B",
        {
            "artifact": "latentport/LPCompact-Qwen3.8-27B",
        },
    )
    transaction.pool = SimpleNamespace(
        get_num_free_blocks=lambda: int(outcome != "no pages"),
        get_new_blocks=lambda count: reserved,
        free_blocks=lambda blocks: frees.extend(block.block_id for block in blocks),
    )
    transaction.managers = [live]
    transaction.cache = SimpleNamespace(get_block_ids=lambda _: ([1, 2, 3, 4],))
    transaction.groups = (0,)
    target = 245 if outcome == "same page count" else 64
    transaction._compact("r", 256, 256, target)

    if outcome == "commit":
        assert [event[0] for event in events] == [
            "latentport_stage",
            "latentport_finish_stage",
        ]
        assert live.req_to_blocks["r"] == reserved
        assert live.packed_rows_dropped == {"r": 192}
        assert frees == [4, 3, 2, 1]
    else:
        assert live.req_to_blocks["r"] == source
        assert live.packed_rows_dropped == {}
        assert frees == ([10] if outcome == "failed copy" else [])
        if outcome == "failed copy":
            assert events[-1] == ("latentport_finish_stage", ("r", None))
        else:
            assert not events
