"""Generate legible print copies from saved results, without changing history."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

if __name__ == '__main__':
    from slm_steering.excellence.print_figures import render
    render(ROOT)
