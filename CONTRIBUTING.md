# Contributing

Start with a focused issue or a small patch. Include a runnable reproduction for fixes.

```powershell
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[gguf,dev]"
python -m pytest -q
```

Lightweight packaging checks also run without inference extras:
`python -m unittest discover -s tests -p test_distribution.py`.

GPU checks are separate and require an idle supported GPU. Use `scripts/smoke_gguf.py`
for state/MTP correctness and `scripts/check_long_chat.py --policy cached_handoff`
for the bounded long-context conversation check. Both accept explicit model paths and
require `--gpu-is-free`. Keep large weights, captured states and build output outside
this repository. Never submit credentials or private conversation snapshots.

Behavioral contracts matter: no silent target replay; exact compatibility checks;
immutable published states; independent branches; rollback before publishing a speculative
state; and honest timing including source and translation work.

Native changes must preserve the pinned ABI or explicitly version it. Follow
[the native guide](docs/native-abi.md). New platforms and bridges need their own
qualification; passing CPU mocks does not establish real-model correctness.
