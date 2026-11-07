"""Recheck reused Qwen source labels under the new isolated evaluation limit."""
from __future__ import annotations

import json

from slm_steering.final_extension.common import ROOT, REPORT, RUN, atomic_json, file_hash, read_json, now, writer_lock
from slm_steering.final_extension.sandbox import SandboxedEvalPlus


def run():
    bank_path = ROOT / 'runs/candidate_bank/definitive_v0.2/trajectories.jsonl'
    bank = [json.loads(line) for line in bank_path.read_text(encoding='utf-8').splitlines()]
    directory = RUN / 'qwen/source_label_audit'
    mismatches, completed = [], 0
    with writer_lock(directory), SandboxedEvalPlus('humaneval') as verifier:
        for task in sorted({r['task_id'] for r in bank}):
            path = directory / (task.replace('/', '_') + '.json')
            if path.exists():
                value = read_json(path)
                if value['source_sha256'] != file_hash(bank_path):
                    raise ValueError('Historical bank changed during label audit')
            else:
                rows = []
                for r in sorted((r for r in bank if r['task_id'] == task), key=lambda r: r['attempt_id']):
                    checked = verifier.verify(task, r['extracted_code'])
                    rows.append({'task_id': task, 'attempt_id': r['attempt_id'], 'historical_passed': r['passed'],
                                 'sandbox_passed': checked['passed'], 'verifier_status': checked['verifier_status'],
                                 'matches': checked['passed'] == r['passed']})
                value = {'source_sha256': file_hash(bank_path), 'task_id': task, 'rows': rows,
                         'benchmark': verifier.dataset['metadata'], 'completed_at': now()}
                atomic_json(path, value)
            completed += len(value['rows'])
            mismatches.extend(r for r in value['rows'] if not r['matches'])
            atomic_json(REPORT / 'source_label_audit_status.json', {'completed': completed, 'expected': len(bank),
                        'mismatches': mismatches, 'updated_at': now()})
            print(f'Qwen sandbox source-label audit {completed}/{len(bank)} mismatches={len(mismatches)}', flush=True)
    result = {'completed_at': now(), 'candidates': completed, 'source_sha256': file_hash(bank_path),
              'mismatches': mismatches, 'all_match': not mismatches, 'memory_limit_bytes': 3221225472,
              'historical_default_memory_limit_bytes': 4294967296,
              'scope': 'Official base+plus checker rerun in OS sandbox; original labels/artifacts were not changed.'}
    atomic_json(REPORT / 'source_label_audit.json', result)
    if mismatches:
        raise ValueError('Reused source labels differ under sandboxed evaluator; document impact before external freeze')
    return result
