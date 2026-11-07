"""Reproduce exploratory diagnostics without changing frozen research artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from research_diagnostics import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports/final_review/supplemental')
    args = parser.parse_args()
    evidence = json.loads((ROOT / 'reports/final_review/evidence.json').read_text(encoding='utf-8'))
    if not evidence.get('confirmatory_results_available'):
        raise ValueError('Complete fixed-design results are required before post-hoc diagnostics')
    result = build(ROOT, args.output_dir.resolve(), evidence)
    print(json.dumps({'status': result['status'], 'primary_reproduction': result['primary_reproduction'],
                      'influence': result['influence_summary'],
                      'retention': {k: v['aggregate'] for k, v in result['correct_retention'].items()}}, indent=2))


if __name__ == '__main__':
    main()
