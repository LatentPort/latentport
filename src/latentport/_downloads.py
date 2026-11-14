"""Checksum-verified runtime downloads with recoverable partial files."""
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen
import json
import zipfile

from .hub import bulk_home


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download(spec):
    cache = bulk_home() / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (spec["sha256"] + ".zip")
    if target.exists() and digest(target) == spec["sha256"]:
        return target
    fd, temporary = tempfile.mkstemp(dir=cache, suffix=".partial")
    try:
        print(f"Downloading {spec.get('name', 'runtime')} ({spec.get('bytes', 'unknown')} bytes)", flush=True)
        with os.fdopen(fd, "wb") as output, urlopen(spec["url"], timeout=60) as response:
            for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                output.write(block)
        if digest(temporary) != spec["sha256"]:
            raise RuntimeError("Runtime archive checksum mismatch")
        Path(temporary).replace(target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def install_windows_cuda():
    """Install only runtime DLLs and their supplied notices, never a driver/toolkit."""
    spec = json.loads((Path(__file__).parent / "configs/cuda-windows.json").read_text())
    destination = bulk_home() / "native/cuda13"
    destination.mkdir(parents=True, exist_ok=True)
    for component in spec["components"]:
        archive_path = download(component)
        with zipfile.ZipFile(archive_path) as archive:
            for entry in archive.infolist():
                name = Path(entry.filename).name
                if not name or entry.is_dir():
                    continue
                is_dll = name.lower().endswith(".dll")
                is_notice = any(word in name.lower() for word in ("license", "notice", "eula"))
                if not (is_dll or is_notice):
                    continue
                output = destination / (name if is_dll else component["name"] + "-" + name)
                fd, temporary = tempfile.mkstemp(dir=destination)
                try:
                    with os.fdopen(fd, "wb") as stream, archive.open(entry) as source:
                        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                            stream.write(block)
                    Path(temporary).replace(output)
                finally:
                    Path(temporary).unlink(missing_ok=True)
    return destination
