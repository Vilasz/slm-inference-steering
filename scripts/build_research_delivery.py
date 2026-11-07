from __future__ import annotations

import argparse
from datetime import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from slm_steering.research.delivery import build_evidence, legacy_reanalysis


def main():
    parser = argparse.ArgumentParser(description='Build auditable TCC evidence from observed local artifacts')
    parser.add_argument('--as-of', default=datetime.now().date().isoformat())
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports/final_review')
    parser.add_argument('--reanalyze-legacy', action='store_true')
    parser.add_argument('--legacy-only', action='store_true')
    parser.add_argument('--allow-running-snapshot', action='store_true', help='Publish a separate stable JSONL prefix while collection continues')
    args = parser.parse_args()
    if args.legacy_only:
        legacy_reanalysis(ROOT, args.output_dir)
    else:
        evidence = build_evidence(ROOT, args.output_dir, as_of=args.as_of, reanalyze=args.reanalyze_legacy,
                                  allow_running_snapshot=args.allow_running_snapshot)
        if evidence.get('confirmatory_results_available'):
            from research_diagnostics import build
            from slm_steering.research.protocol import file_hash, write_json
            evidence['supplemental'] = build(ROOT, args.output_dir.resolve() / 'supplemental', evidence)
            evidence['inputs'].update(evidence['supplemental']['inputs'])
            evidence['inputs'].update(evidence['supplemental']['artifacts'])
            excellence_path = ROOT / 'reports/excellence/results.json'
            if excellence_path.exists():
                import json
                excellence = json.loads(excellence_path.read_text(encoding='utf-8'))
                for relative, expected in excellence['inputs'].items():
                    if file_hash(ROOT / relative) != expected:
                        raise ValueError(f'Excellence synthesis input changed; rebuild the report: {relative}')
                evidence['excellence'] = excellence
                evidence['inputs'].update(excellence['inputs'])
                evidence['inputs']['reports/excellence/results.json'] = file_hash(excellence_path)
                materials_path = ROOT / 'reports/excellence/materials/manifest.json'
                materials = json.loads(materials_path.read_text(encoding='utf-8'))
                for relative, expected in {**materials['inputs'], **materials['artifacts']}.items():
                    if file_hash(ROOT / relative) != expected:
                        raise ValueError(f'Academic materials changed or are stale; rebuild them: {relative}')
                    evidence['inputs'][relative] = expected
                evidence['inputs']['reports/excellence/materials/manifest.json'] = file_hash(materials_path)
                evidence['excellence_materials'] = materials
            for name in ('research_diagnostics.py', 'analyze_research_robustness.py',
                         'build_research_delivery.py', 'build_final_notebook.py', 'render_main_manuscript.py',
                         'supplemental_publication.py', 'build_manuscript.py', 'audit_research_delivery.py',
                         'update_execution_summary.py'):
                relative = f'scripts/{name}'
                evidence['inputs'][relative] = file_hash(ROOT / relative)
            write_json(args.output_dir / 'evidence.json', evidence)
        from publication_visuals import render
        render(ROOT, args.output_dir, evidence)
    print(f'Evidence written to {args.output_dir}')


if __name__ == '__main__':
    main()
