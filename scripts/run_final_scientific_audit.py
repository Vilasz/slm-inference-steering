"""Reproduce final numerical tables without launching model generation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

if __name__ == '__main__':
    from slm_steering.excellence.final_audit import run
    run(ROOT)
