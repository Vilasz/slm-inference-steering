from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slm_steering.env import collect_torch_environment, cuda_install_hint, format_torch_environment


def main() -> None:
    env = collect_torch_environment()
    print(format_torch_environment(env))
    if not env.cuda_available:
        print("\nCUDA indisponivel neste ambiente Python.\n")
        print(cuda_install_hint())
        raise SystemExit(1)
    print("\nCUDA ok para este ambiente Python.")


if __name__ == "__main__":
    main()
