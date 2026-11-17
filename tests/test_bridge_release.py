import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from latentport.hub import bundle_root, DEFAULT_REPO
from latentport._reference.bridge import ReferenceTranslator
from latentport.port import Port
from test_runtime import pair


def make_bundle(root, repo, width):
    root.mkdir()
    configs = root / "configs"
    configs.mkdir()
    (configs / "geometry.json").write_text(json.dumps({"source": {"hidden": width}, "target": {"hidden": 5120}}))
    for name in ("calibration", "architecture_v2"):
        (configs / (name + ".json")).write_text("{}")
    (root / "bridge_config.json").write_text(json.dumps({
        "architecture": "qwen-hybrid-mixed-v1", "weights": {"parent": "bridge.safetensors"}}))
    (root / "bridge.safetensors").write_bytes(repo.encode())
    (root / "bridge.gguf").write_bytes(b"GGUF" + repo.encode())
    files = {p.relative_to(root).as_posix(): {"bytes":p.stat().st_size, "sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in root.rglob("*") if p.is_file()}
    endpoint = {"repo_id": "model/source", "revision": "1"*40, "metadata_files": {}, "weight_files": {}}
    (root / "latentport.json").write_text(json.dumps({
        "schema": "latentport.bridge-bundle.v2", "repo_id": repo, "architecture": "qwen-hybrid-mixed-v1",
        "gguf_bridge": "bridge.gguf", "artifacts": files, "models": {"source": endpoint, "target": endpoint}}))
    return root


def test_two_routes_resolve_their_own_revision_weights_and_geometry(tmp_path):
    first = make_bundle(tmp_path / ("1"*40), DEFAULT_REPO, 2560)
    second = make_bundle(tmp_path / ("2"*40), "fixtures/9b-route", 4096)
    paths = {DEFAULT_REPO:first, "fixtures/9b-route":second}
    with patch("latentport.hub._snapshot", side_effect=lambda repo, revision, patterns:paths[repo]) as fetch:
        a = ReferenceTranslator(bundle_root(DEFAULT_REPO, full=True, backend="gguf"))
        b = ReferenceTranslator(bundle_root("fixtures/9b-route", revision="2"*40, full=True, backend="nf4"))
        assert (a.geo["source"]["hidden"], b.geo["source"]["hidden"]) == (2560,4096)
        second_request = fetch.call_args_list[3]
        assert second_request.args == ("fixtures/9b-route", "2"*40)
        assert "bridge.safetensors" in second_request.kwargs["patterns"]
        assert "bridge.gguf" not in second_request.kwargs["patterns"]
    (second / "bridge.safetensors").write_bytes(b"x" * len("fixtures/9b-route"))
    with pytest.raises(ValueError, match="SHA-256"):
        bundle_root(second, full=True, backend="nf4")


def test_query_handoff_consumes_exact_suffix_once(tmp_path):
    source, target, bridge, backend = pair(tmp_path)
    prefix = source.capture("prefix")
    result = Port(source, target, bridge).handoff(prefix, target_suffix_ids=[65, 66])
    assert result.target_prompt_tokens == 2
    assert result.clock == prefix.clock + 2
    assert result.phase == "answer_ready"
    assert backend.calls[-1][-1] == (65, 66)


def test_gguf_rejects_model_shards_in_tokenizer_metadata(tmp_path):
    root = make_bundle(tmp_path / "pair", DEFAULT_REPO, 2560)
    path = root / "latentport.json"
    manifest = json.loads(path.read_text())
    manifest["models"]["source"]["metadata_files"]["model-00001-of-00003.safetensors"] = {}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="tokenizer metadata"):
        bundle_root(root, full=True, backend="gguf")


def test_local_base_model_cannot_shadow_or_omit_a_pinned_shard(tmp_path):
    from latentport.hub import verify_model_weights
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map":{"x":"part.safetensors"}}))
    spec = {"weight_files":{index.name:{"bytes":index.stat().st_size,"sha256":hashlib.sha256(index.read_bytes()).hexdigest()}}}
    with pytest.raises(ValueError, match="inventory"):
        verify_model_weights(tmp_path,spec)
    (tmp_path / "model.safetensors").touch()
    with pytest.raises(ValueError, match="shadows"):
        verify_model_weights(tmp_path,spec)


def test_layer_aligned_nf4_bundle_without_gguf(tmp_path):
    from latentport.errors import CapabilityError
    from latentport.hub import LAYER_ALIGNED

    root = make_bundle(tmp_path / "layer-aligned", "fixtures/layer-aligned", 4096)
    manifest_path = root / "latentport.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["architecture"] = LAYER_ALIGNED
    del manifest["gguf_bridge"]
    for name in ("bridge.gguf", "configs/architecture_v2.json", "configs/calibration.json"):
        del manifest["artifacts"][name]
        (root / name).unlink()
    config = root / "bridge_config.json"
    config.write_text(json.dumps({"architecture": LAYER_ALIGNED, "weights": {"writer": "bridge.safetensors"}}))
    manifest["artifacts"][config.name] = {
        "bytes": config.stat().st_size, "sha256": hashlib.sha256(config.read_bytes()).hexdigest()}
    manifest_path.write_text(json.dumps(manifest))
    assert bundle_root(root, full=True, backend="nf4") == root
    for backend in ("bf16", "gguf"):
        with pytest.raises(CapabilityError, match="NF4"):
            bundle_root(root, full=True, backend=backend)
