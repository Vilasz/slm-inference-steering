from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class TorchEnvironment:
    python_executable: str
    python_version: str
    platform: str
    torch_installed: bool
    torch_version: str | None
    torch_cuda_version: str | None
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    nvidia_smi: str | None


def collect_torch_environment() -> TorchEnvironment:
    torch_installed = False
    torch_version = None
    torch_cuda_version = None
    cuda_available = False
    cuda_device_count = 0
    cuda_device_name = None

    try:
        import torch

        torch_installed = True
        torch_version = torch.__version__
        torch_cuda_version = torch.version.cuda
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count()
        if cuda_available:
            cuda_device_name = torch.cuda.get_device_name(0)
    except Exception as exc:
        torch_version = f"import failed: {exc}"

    return TorchEnvironment(
        python_executable=sys.executable,
        python_version=platform.python_version(),
        platform=platform.platform(),
        torch_installed=torch_installed,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_name=cuda_device_name,
        nvidia_smi=_query_nvidia_smi(),
    )


def format_torch_environment(env: TorchEnvironment) -> str:
    lines = [
        f"python_executable: {env.python_executable}",
        f"python_version: {env.python_version}",
        f"platform: {env.platform}",
        f"torch_installed: {env.torch_installed}",
        f"torch_version: {env.torch_version}",
        f"torch_cuda_version: {env.torch_cuda_version}",
        f"cuda_available: {env.cuda_available}",
        f"cuda_device_count: {env.cuda_device_count}",
        f"cuda_device_name: {env.cuda_device_name}",
    ]
    if env.nvidia_smi:
        lines.append("nvidia_smi:")
        lines.append(env.nvidia_smi)
    else:
        lines.append("nvidia_smi: unavailable")
    return "\n".join(lines)


def cuda_install_hint() -> str:
    return (
        "No Windows, use Python 3.10-3.12 para PyTorch com CUDA. "
        "Recrie a venv com Python 3.12 e instale o wheel CUDA, por exemplo:\n\n"
        "  Remove-Item -Recurse -Force .venv\n"
        "  uv venv --python 3.12\n"
        "  uv pip install -r requirements.txt\n\n"
        "Depois valide com:\n\n"
        "  python scripts/diagnose_cuda.py\n\n"
        "O requirements.txt fixa torch==2.12.0+cu126 pelo indice CUDA. "
        "Para este projeto de LLM, instale apenas torch; torchvision/torchaudio "
        "nao sao necessarios. Para a RTX 3050 de 6 GB, mantenha o baseline em dtype automatico "
        "(float16 em CUDA) ou passe --dtype float16 explicitamente.\n"
    )


def assert_cuda_available() -> None:
    env = collect_torch_environment()
    if env.cuda_available:
        return
    raise RuntimeError(
        "CUDA foi solicitado, mas torch.cuda.is_available() retornou False.\n\n"
        f"{format_torch_environment(env)}\n\n"
        f"{cuda_install_hint()}"
    )


def _query_nvidia_smi() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()
