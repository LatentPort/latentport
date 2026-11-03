"""Load native dependencies and inspect the ABI; never initialize a model."""
import ctypes
import json
from latentport._native.engine import Library, Field, Info, Receipt, library_path

library=Library(None)
# Fails before backend/model initialization, exercising the C error boundary.
assert not library.lib.lp_open(b"",2,32,16,1,0)
assert library.lib.lp_error()==b"invalid model side"
print(json.dumps({"library":str(library_path()),"abi":library.lib.lp_abi(),
                  "field_bytes":ctypes.sizeof(Field),"info_bytes":ctypes.sizeof(Info),
                  "receipt_bytes":ctypes.sizeof(Receipt),"models_initialized":False}))
