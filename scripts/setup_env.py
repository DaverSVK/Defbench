"""Create an isolated environment from configs/environments.yaml.

    python scripts/setup_env.py core
    python scripts/setup_env.py ultralytics rfdetr detr_family
    python scripts/setup_env.py detectron2      # builds the Docker image (Linux / WSL2 + NVIDIA toolkit)

venv environments are created with `uv` if available (fast, shared cache), otherwise with
`python -m venv` + pip. The CUDA wheel of torch is installed first (index URL from the config).
"""
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sh(cmd):
    print("+", " ".join(map(str, cmd)))
    subprocess.check_call([str(c) for c in cmd], cwd=ROOT)


def py_of(path: Path) -> Path:
    return path / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")


def setup_venv(name, e):
    path = ROOT / e["path"]
    uv = shutil.which("uv")
    torch_spec = e["torch"].split()
    if uv:
        if not py_of(path).exists():
            sh([uv, "venv", path, "--python", e.get("python", "3.11")])
        pip = [uv, "pip", "install", "-p", py_of(path)]
    else:
        if not py_of(path).exists():
            sh([sys.executable, "-m", "venv", path])
        pip = [py_of(path), "-m", "pip", "install"]
    sh(pip + torch_spec)
    sh(pip + ["-r", ROOT / e["requirements"]])
    sh([py_of(path), "-c", "import torch;print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"])


def setup_docker(name, e):
    if not shutil.which("docker"):
        raise SystemExit("docker is required for environment " + name)
    # build context = docker/ (the Dockerfiles do not COPY project files; the project is mounted at run time)
    sh(["docker", "build", "-f", ROOT / e["dockerfile"], "-t", e["image"], ROOT / "docker"])


def main(names):
    envs = yaml.safe_load((ROOT / "configs" / "environments.yaml").read_text(encoding="utf-8"))
    for n in names or ["core"]:
        e = envs[n]
        (setup_venv if e["type"] == "venv" else setup_docker)(n, e)


if __name__ == "__main__":
    main(sys.argv[1:])
