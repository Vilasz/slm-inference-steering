"""Run the bounded known-task benchmark only after independent collection."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    import torch  # initialize CUDA DLLs before scipy/sklearn on Windows
    from slm_steering.excellence.operational import run
    _ = torch.__version__
    run(ROOT)


if __name__ == '__main__':
    main()
