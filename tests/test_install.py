"""Release downloads must validate content before installing native code."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from latentport.__main__ import install_native
from latentport.hub import gguf_model_file


class InstallTests(unittest.TestCase):
    def test_gguf_accepts_extensionless_hub_blob(self):
        from latentport.gguf import GGUFBackend
        with tempfile.TemporaryDirectory() as tmp:
            blob = Path(tmp)/("a"*64)
            blob.write_bytes(b"GGUF")
            backend = GGUFBackend(blob, blob, Path(tmp), {})
            self.assertEqual(backend.paths["source"], blob.resolve())

    def test_gguf_rejects_non_gguf_content(self):
        from latentport.gguf import GGUFBackend
        from latentport.errors import CompatibilityError
        with tempfile.TemporaryDirectory() as tmp:
            blob = Path(tmp)/"model.gguf"
            blob.write_bytes(b"HTML")
            with self.assertRaises(CompatibilityError):
                GGUFBackend(blob, blob, Path(tmp), {})

    def test_model_checksum_rejects_changed_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"model.gguf"
            p.write_bytes(b"modified")
            spec = {"source": {"repo_id": "test/model", "revision": "pinned", "filename": p.name,
                                "bytes": 8, "sha256": hashlib.sha256(b"original").hexdigest()}}
            with patch.object(Path, "read_text", return_value=json.dumps(spec)), \
                 patch("huggingface_hub.hf_hub_download", return_value=str(p)):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    gguf_model_file("source")

    def _native(self, member, bad_hash=False):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)/"cache"
            p = Path(tmp)/"native.zip"
            with zipfile.ZipFile(p, "w") as z:
                z.writestr(member, b"test-binary")
            spec = {"repo_id": "test/native", "filename": p.name, "revision": "pinned",
                    "sha256": "0"*64 if bad_hash else hashlib.sha256(p.read_bytes()).hexdigest()}
            with patch.object(Path, "read_text", return_value=json.dumps(spec)), \
                 patch("huggingface_hub.hf_hub_download", return_value=str(p)), \
                 patch("latentport.__main__.bulk_home", return_value=home), \
                 patch("platform.system", return_value="Windows"), \
                 patch("platform.machine", return_value="AMD64"):
                if bad_hash or member != "llama.dll":
                    with self.assertRaises(RuntimeError):
                        install_native()
                    self.assertFalse((home/"native/build/bin/llama.dll").exists())
                    self.assertFalse((home/"native/build/escape.dll").exists())
                else:
                    install_native()
                    self.assertEqual((home/"native/build/bin/llama.dll").read_bytes(), b"test-binary")

    def test_verified_native_archive_installs(self):
        self._native("llama.dll")

    def test_native_checksum_failure_installs_nothing(self):
        self._native("llama.dll", bad_hash=True)

    def test_native_archive_cannot_escape_destination(self):
        self._native("../escape.dll")

    def test_linux_release_download_is_verified_and_reused(self):
        import io
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp)/"cache"
            data=io.BytesIO()
            with zipfile.ZipFile(data,"w") as archive:
                archive.writestr("libllama.so",b"qualified-native")
            raw=data.getvalue()
            spec={"linux":{"url":"https://github.com/LatentPort/latentport/releases/download/test/native.zip",
                           "sha256":hashlib.sha256(raw).hexdigest()}}
            with patch.object(Path,"read_text",return_value=json.dumps(spec)), \
                 patch("latentport.__main__.bulk_home",return_value=home), \
                 patch("platform.system",return_value="Linux"), \
                 patch("platform.machine",return_value="x86_64"), \
                 patch("latentport.__main__.urlopen",return_value=io.BytesIO(raw)) as download:
                install_native()
                install_native()
                self.assertEqual(download.call_count,1)
                self.assertEqual((home/"native/build/bin/libllama.so").read_bytes(),b"qualified-native")

    def test_release_download_rejects_corrupted_binary_before_install(self):
        import io
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp)/"cache"
            spec={"url":"https://github.com/LatentPort/latentport/releases/download/test/native.zip","sha256":"0"*64}
            with patch.object(Path,"read_text",return_value=json.dumps(spec)), \
                 patch("latentport.__main__.bulk_home",return_value=home), \
                 patch("platform.system",return_value="Windows"), \
                 patch("platform.machine",return_value="AMD64"), \
                 patch("latentport.__main__.urlopen",return_value=io.BytesIO(b"changed")):
                with self.assertRaisesRegex(RuntimeError,"checksum"):
                    install_native()
                self.assertFalse((home/"native/build/bin").exists())
                self.assertEqual(list((home/"downloads").iterdir()),[])
