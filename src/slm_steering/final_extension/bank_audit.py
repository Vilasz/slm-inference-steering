"""Independent provenance, seed, causal-shape and risk-set checks on new banks."""
from __future__ import annotations

import numpy as np

from slm_steering.final_extension.collection import records
from slm_steering.final_extension.common import ROOT, RUN, REPORT, atomic_json, now, read_json
from slm_steering.final_extension.preparation import MODELS
from slm_steering.research.protocol import attempt_seed


def audit_record(record, layers):
    name = record['model']
    expected_seed = attempt_seed('final-cross-model-v1-' + name, record['benchmark_version'],
                                 record['task_id'], record['attempt_id'])
    if record['seed'] != expected_seed:
        raise ValueError('Seed/identity provenance differs')
    if record['model_revision'] != MODELS[name]['revision'] or record['tokenizer_revision'] != MODELS[name]['revision']:
        raise ValueError('Model/tokenizer revision differs')
    if record['generated_tokens'] != len(record['token_ids']) or not 1 <= record['generated_tokens'] <= 512:
        raise ValueError('Completion length differs')
    for key in ('token_logprobs', 'token_entropies', 'token_margins'):
        values = np.asarray(record[key], float)
        if len(values) != min(record['generated_tokens'], 32) or not np.isfinite(values).all():
            raise ValueError('Causal observable shape/nonfiniteness')
    if record['passed'] != all(s == 'pass' for s in record['verifier_status'].values()):
        raise ValueError('Label differs from base+plus statuses')
    if layers != [17 if name == 'deepseek' else 20]:
        raise ValueError('Relative-depth layer identity differs')
    return True


def audit(phase='development'):
    results = []
    phases = ('pilot', 'technical', 'source') if phase == 'development' else ('external',)
    for name in ('deepseek', 'qwen'):
        for label in phases:
            directory = RUN / name / label
            if not (directory / 'manifest.json').exists():
                continue
            manifest = read_json(directory / 'manifest.json')
            reference = (read_json(ROOT / 'runs/candidate_bank/definitive_v0.2/manifest.json')['benchmark']
                         if label != 'external' else read_json(REPORT / 'task_inventory.json')['dataset'])
            for field in ('version', 'dataset_sha256', 'evalplus_version', 'official_dataset_md5'):
                if manifest['benchmark'][field] != reference[field]:
                    raise ValueError(f'Benchmark identity differs: {name}/{label}/{field}')
            if not manifest.get('numerical_checks') or not all(c['passed'] for c in manifest['numerical_checks']):
                raise ValueError('Real-model causal/distribution checks were not completed')
            bank = records(directory)
            if manifest['status'] != 'completed' or len(bank) != manifest['expected_attempts']:
                raise ValueError(f'Incomplete bank cannot pass publication audit: {name}/{label}')
            for r in bank:
                with np.load(directory / r['activation_path'], allow_pickle=False) as arrays:
                    audit_record(r, arrays['layers'].tolist())
                    if r['generated_tokens'] > 32 and '32' not in arrays:
                        raise ValueError('Missing active checkpoint')
                    if any(not np.isfinite(arrays[k]).all() for k in arrays.files):
                        raise ValueError('Nonfinite activation')
            results.append({'model': name, 'phase': label, 'candidates': len(bank),
                            'tasks': len({r['task_id'] for r in bank}), 'all_checks_passed': True})
    value = {'audited_at': now(), 'phase': phase, 'banks': results}
    atomic_json(REPORT / f'{phase}_bank_audit.json', value)
    return value
