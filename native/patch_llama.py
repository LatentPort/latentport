"""Apply the minimal LatentPort hooks to a disposable, pinned llama.cpp tree."""
from pathlib import Path
import subprocess
import sys

PIN = "b19cbe925be361d229f0fe03435affe4a2717f37"
root = Path(sys.argv[1]).resolve()
head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
if head != PIN:
    raise SystemExit(f"Expected llama.cpp {PIN}, found {head}")

def edit(name, old, new):
    path = root / name
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if text.count(old) != 1:
        raise SystemExit(f"Patch anchor changed: {name}: {old[:70]!r}")
    path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")

# Internal typed access is isolated to this pinned native build.
edit("src/llama-kv-cache.h", "class llama_kv_cache : public llama_memory_i {",
     "class llama_kv_cache : public llama_memory_i {\n    friend struct latentport_access;")
edit("src/models/qwen35.cpp", '#include "models.h"', '#include "models.h"\n#include "latentport.h"')
edit("src/models/qwen35.cpp", '    cb(qkv_mixed, "linear_attn_qkv_mixed", il);',
     '    cb(qkv_mixed, "linear_attn_qkv_mixed", il);\n    lp_graph_tap(cparams.cb_eval_user_data, ctx0, gf, qkv_mixed, 3, il);')
edit("src/models/qwen35.cpp", '        cb(cur, "l_out", il);',
     '        cb(cur, "l_out", il);\n        if (il == n_layer - 1) lp_graph_tap(cparams.cb_eval_user_data, ctx0, gf, cur, 0, il);')
edit("src/models/qwen35.cpp", '    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN35 MTP currently only supports a single MTP block");',
     '    GGML_ASSERT(cparams.nextn_layer_offset >= 0 && cparams.nextn_layer_offset < (int) hparams.n_layer_nextn);')
edit("src/models/qwen35.cpp", '    const int il = hparams.n_layer();',
     '    const int il = hparams.n_layer() + cparams.nextn_layer_offset;')
print(f"Patched isolated llama.cpp {PIN}")
