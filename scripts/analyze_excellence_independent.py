"""Audit the complete independent sample without changing its frozen endpoint."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

if __name__ == '__main__':
    from slm_steering.excellence.independent_report import run
    run(ROOT)
