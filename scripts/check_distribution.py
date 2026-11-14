"""Smoke-test the installed wheel from outside the repository, without extras."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


with tempfile.TemporaryDirectory() as directory:
    env = dict(os.environ, LATENTPORT_HOME=str(Path(directory) / "cache"))
    env.pop("PYTHONPATH", None)
    code = '''
from importlib.resources import files
import latentport
assert files('latentport').joinpath('_demo/web/index.html').is_file()
assert files('latentport').joinpath('_demo/web/app.js').is_file()
assert files('latentport').joinpath('configs/native-release.json').is_file()
'''
    subprocess.run([sys.executable, "-c", code], env=env, cwd=directory, check=True)
    subprocess.run([sys.executable, "-m", "latentport", "demo", "--help"], env=env, cwd=directory, check=True)
    result = subprocess.run([sys.executable, "-m", "latentport", "doctor"], env=env,
                            cwd=directory, capture_output=True, text=True)
    import json
    report = json.loads(result.stdout)
    assert report["package_installed"] and "missing_dependencies" in report
    assert result.returncode == (0 if report["ready_for_model_load"] else 1)
    assert not (Path(directory) / "cache").exists(), "Read-only commands created a model cache"
print("Installed distribution checks passed")
