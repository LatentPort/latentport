"""CPU-only byte and shape comparison of the bridge GGUF against its release map."""
import argparse
import hashlib
import json
from pathlib import Path
from latentport._native.gguf_reader import BridgeGGUF

parser=argparse.ArgumentParser()
parser.add_argument("gguf",type=Path)
parser.add_argument("--tensor-map",type=Path,required=True)
args=parser.parse_args()
mapping=json.loads(args.tensor_map.read_text(encoding="utf-8"))
reader=BridgeGGUF(args.gguf)
if set(reader.tensors)!=set(mapping):raise SystemExit("Tensor names differ")
for name,array in reader.tensors.items():
    record=mapping[name]
    if tuple(array.shape)!=tuple(record["canonical_shape"]):raise SystemExit(f"Shape differs: {name}")
    actual=hashlib.sha256(memoryview(array).cast("B")).hexdigest()
    if actual!=record["payload_sha256"]:raise SystemExit(f"Payload differs: {name}")
print(json.dumps({"tensor_count":len(mapping),"all_shapes_and_payloads_match":True,"gpu_execution":False}))
