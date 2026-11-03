"""Small, bounded GGUF v3 reader for the released F32 translator container.

LM loading and quantized tensor execution belong to llama.cpp. This reader
deliberately rejects quantized bridge tensors rather than guessing a format.
"""
from __future__ import annotations
import math
from pathlib import Path
import struct
import numpy as np
from ..errors import CompatibilityError


class BridgeGGUF:
    def __init__(self, path):
        self.path = Path(path)
        self.size = self.path.stat().st_size
        with self.path.open("rb") as f:
            self._f = f
            if self._read(4) != b"GGUF" or self._unpack("I") != 3:
                raise CompatibilityError("Bridge requires little-endian GGUF v3")
            nt, nk = self._unpack("Q"), self._unpack("Q")
            if nt > 4096 or nk > 4096:
                raise CompatibilityError("Unreasonable GGUF header counts")
            self.metadata = {}
            for _ in range(nk):
                key = self._string()
                if key in self.metadata:
                    raise CompatibilityError("Duplicate GGUF metadata")
                self.metadata[key] = self._value(self._unpack("I"))
            entries = []
            for _ in range(nt):
                name, ndim = self._string(), self._unpack("I")
                if not 1 <= ndim <= 4:
                    raise CompatibilityError("Unsupported bridge tensor rank")
                dims = tuple(self._unpack("Q") for _ in range(ndim))
                typ, offset = self._unpack("I"), self._unpack("Q")
                if typ != 0 or any(d < 1 for d in dims):
                    raise CompatibilityError("Bridge supports F32 tensors only")
                entries.append((name, tuple(reversed(dims)), offset))
            alignment = self.metadata.get("general.alignment", 32)
            if not isinstance(alignment, int) or alignment < 1 or alignment > 4096 or alignment & (alignment-1):
                raise CompatibilityError("Invalid GGUF alignment")
            start = (f.tell() + alignment-1) // alignment * alignment
        if self.metadata.get("general.architecture") != "latentport_bridge_v1":
            raise CompatibilityError("Not a LatentPort bridge container")
        self.tensors = {}
        ranges = []
        for name, shape, offset in entries:
            length = math.prod(shape)*4
            if name in self.tensors or offset % alignment or start+offset+length > self.size:
                raise CompatibilityError("Invalid bridge tensor storage")
            ranges.append((offset, offset+length))
            self.tensors[name] = np.memmap(self.path, mode="c", dtype="<f4", offset=start+offset, shape=shape)
            # GGUF represents this canonical scalar as a one-element vector.
            if name == "parent.base.terminal_gain":
                if shape != (1,):
                    raise CompatibilityError("Terminal gain shape changed")
                self.tensors[name] = self.tensors[name].reshape(())
        ranges.sort()
        if any(b[0] < a[1] for a,b in zip(ranges, ranges[1:])):
            raise CompatibilityError("Overlapping bridge tensor storage")

    def _read(self, n):
        if n < 0 or n > self.size-self._f.tell():
            raise CompatibilityError("Truncated GGUF")
        value = self._f.read(n)
        if len(value) != n:
            raise CompatibilityError("Truncated GGUF")
        return value

    def _unpack(self, fmt):
        return struct.unpack("<"+fmt, self._read(struct.calcsize(fmt)))[0]

    def _string(self):
        n = self._unpack("Q")
        if n > 16*1024*1024:
            raise CompatibilityError("GGUF string too large")
        return self._read(n).decode("utf-8")

    def _value(self, typ, depth=0):
        formats = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}
        if typ in formats:
            return self._unpack(formats[typ])
        if typ == 8:
            return self._string()
        if typ == 9 and depth == 0:
            sub, n = self._unpack("I"), self._unpack("Q")
            if n > 100000:
                raise CompatibilityError("GGUF array too large")
            return [self._value(sub, depth+1) for _ in range(n)]
        raise CompatibilityError("Unsupported GGUF metadata type")
