"""Install the pinned upstream runtime used by Flash-Next GGUF."""
from pathlib import Path
import platform
import tarfile
import tempfile
import zipfile

from .._downloads import download
from ..errors import CapabilityError
from ..hub import bulk_home


_BASE = "https://github.com/ggml-org/llama.cpp/releases/download/b12503/"
_PACKAGES = {
    "Windows": ("llama-b12503-bin-win-cpu-x64.zip", 25521102,
                "64bee9b6fece8e66c665c40290cbb84706af23c670041d024b56a71a39859d77"),
    "Linux": ("llama-b12503-bin-ubuntu-x64.tar.gz", 23784463,
              "0d585f8a78ffdfdc4f6303a089aaa1379c15323b33483b6548c85c774a376a9d"),
}


def flash_library_path(options):
    if options.library is not None:
        return Path(options.library).expanduser().resolve(strict=True)
    system = platform.system()
    if system not in _PACKAGES or platform.machine().lower() not in ("amd64", "x86_64"):
        raise CapabilityError("Flash-Next native GGUF distributions are available for Windows and Linux x64")
    filename, length, checksum = _PACKAGES[system]
    library = "llama.dll" if system == "Windows" else "libllama.so"
    parent = bulk_home() / "native" / "flash-b12503"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / checksum
    present = tuple(destination.rglob(library))
    if len(present) == 1 and present[0].is_file():
        return present[0]
    archive = download({"url": _BASE + filename, "name": filename, "bytes": length, "sha256": checksum})
    if archive.stat().st_size != length:
        raise CapabilityError("Flash-Next runtime archive has the wrong size")
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        stage = Path(temporary) / "runtime"
        stage.mkdir()
        if system == "Windows":
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(stage)
        else:
            with tarfile.open(archive, "r:gz") as bundle:
                bundle.extractall(stage, filter="data")
        files = tuple(stage.rglob(library))
        if len(files) != 1 or not files[0].is_file():
            raise CapabilityError("The pinned Flash-Next distribution lacks its main library")
        relative = files[0].relative_to(stage)
        try:
            stage.rename(destination)
        except OSError:
            if not (destination / relative).is_file():
                raise
    return destination / relative
