"""Installation and demo contracts that run without any inference dependencies."""
import argparse
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import urlopen

from latentport._demo.server import App, add_arguments, handler, run
from latentport.diagnostics import diagnose
from latentport.hub import gguf_model_specs
from latentport.errors import CapabilityError


class DistributionTests(unittest.TestCase):
    def test_missing_backend_fails_before_network(self):
        import latentport
        with patch('latentport._dependencies.find_spec', return_value=None), \
             patch('latentport.port.bundle_root') as download:
            with self.assertRaisesRegex(CapabilityError, r'latentport\[gguf\]'):
                latentport.load(backend='gguf')
            download.assert_not_called()

    def test_doctor_without_torch_is_structured(self):
        with patch('latentport.diagnostics.missing_dependencies', return_value=['torch']), \
             patch('latentport.diagnostics.native_spec') as native:
            result = diagnose('bf16')
        self.assertFalse(result['ready_for_model_load'])
        self.assertEqual(result['missing_dependencies'], ['torch'])
        native.assert_not_called()

    def test_driver_floor_is_checked_without_loading_models(self):
        from types import SimpleNamespace
        fake = SimpleNamespace(__version__='test', version=SimpleNamespace(cuda='12.8'),
            cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: 'test GPU',
                get_device_capability=lambda _: (12, 0), mem_get_info=lambda _: (32 * 1024**3, 32 * 1024**3)))
        with patch.dict('sys.modules', {'torch': fake}), \
             patch('latentport.diagnostics.missing_dependencies', return_value=[]), \
             patch('latentport.diagnostics.native_spec', return_value={'cuda': '13.0', 'compute_capability': '12.0'}), \
             patch('latentport.diagnostics.driver_cuda_version', return_value=12080):
            result = diagnose('gguf', check_native=False)
            self.assertFalse(result['ready_for_model_load'])
            self.assertTrue(any('newer NVIDIA driver' in error for error in result['errors']))
            # BF16 does not depend on the GGUF binary's CUDA-13 requirement.
            self.assertTrue(diagnose('bf16')['ready_for_model_load'])

    def test_external_profile_requires_immutable_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'models.json'
            spec = dict(repo_id='upstream/model', filename='model.gguf', revision='main',
                        sha256='a' * 64, bytes=10)
            path.write_text(json.dumps(dict(source=spec, target=spec)))
            with self.assertRaisesRegex(ValueError, 'commit SHA'):
                gguf_model_specs(path)
            spec['revision'] = 'b' * 40
            path.write_text(json.dumps(dict(source=spec, target=spec)))
            self.assertEqual(gguf_model_specs(path)['source']['repo_id'], 'upstream/model')

    def test_ui_only_serves_packaged_assets_and_state(self):
        from http.server import ThreadingHTTPServer
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        args = parser.parse_args(['--ui-only'])
        with tempfile.TemporaryDirectory() as directory:
            app = App(Path(directory), args)
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler(app))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                root = f'http://127.0.0.1:{server.server_port}'
                for resource in ('/', '/app.js', '/style.css', '/logo.png'):
                    with urlopen(root + resource) as response:
                        self.assertEqual(response.status, 200)
                        self.assertGreater(len(response.read()), 0)
                with urlopen(root + '/api/state') as response:
                    self.assertIn('status', json.load(response))
            finally:
                server.shutdown()
                server.server_close()
                worker.join()

    def test_preview_cannot_wait_forever_in_headless_mode(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            run(parser.parse_args(['--ui-only', '--headless']))

    def test_insufficient_disk_fails_before_download(self):
        from latentport.hub import check_download_space
        from types import SimpleNamespace
        with patch('latentport.hub.shutil.disk_usage', return_value=SimpleNamespace(free=100)):
            with self.assertRaisesRegex(RuntimeError, 'Insufficient disk'):
                check_download_space(1000)

    def test_windows_runtime_extracts_dlls_and_notices_only(self):
        from latentport._downloads import install_windows_cuda
        import zipfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / 'cuda.zip'
            with zipfile.ZipFile(artifact, 'w') as archive:
                archive.writestr('cuda/bin/cudart64_13.dll', b'library')
                archive.writestr('cuda/LICENSE', b'notice')
                archive.writestr('cuda/include/cuda.h', b'header')
            with patch('latentport._downloads.bulk_home', return_value=root), \
                 patch('latentport._downloads.download', return_value=artifact):
                output = install_windows_cuda()
            self.assertEqual((output / 'cudart64_13.dll').read_bytes(), b'library')
            self.assertEqual((output / 'cuda_cudart-LICENSE').read_bytes(), b'notice')
            self.assertFalse((output / 'cuda.h').exists())


if __name__ == '__main__':
    unittest.main()
