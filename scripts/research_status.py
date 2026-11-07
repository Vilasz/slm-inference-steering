"""Read current progress without loading a model or changing the experiment."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    runner = ROOT / 'reports/research/full_execution_v0.2/status.json'
    if runner.exists():
        state = json.loads(runner.read_text(encoding='utf-8'))
        print(f"Pipeline: {state['status']} | etapa: {state.get('current_stage')} | PID: {state.get('pid')}")
        if state.get('failure'):
            print(json.dumps(state['failure'], ensure_ascii=False))
    for phase, expected in [('pilot', 960), ('definitive', 2640), ('transfer', 640)]:
        path = ROOT / f'runs/candidate_bank/{phase}_v0.2/manifest.json'
        if not path.exists():
            print(f'{phase}: 0/{expected} | aguardando')
            continue
        manifest = json.loads(path.read_text(encoding='utf-8'))
        count = manifest.get('completed_attempts', 0)
        print(f"{phase}: {count}/{manifest['expected_attempts']} | {manifest['status']}")
    print('Logs: reports/research/full_execution_v0.2/; PDF atualizado após as análises completas.')


if __name__ == '__main__':
    main()
