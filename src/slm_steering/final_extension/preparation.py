from __future__ import annotations

from difflib import SequenceMatcher
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess

import numpy as np
import pandas as pd

from slm_steering.final_extension.common import REPORT, ROOT, atomic_json, assert_history, history_snapshot, read_json
from slm_steering.research.protocol import digest, file_hash, now, attempt_seed
from slm_steering.research.statistics import task_auc, paired_task_bootstrap

MODELS = {
    'qwen': {'id': 'Qwen/Qwen2.5-Coder-1.5B-Instruct', 'revision': '2e1fd397ee46e1388853d2af2c993145b0f1098a'},
    'deepseek': {'id': 'deepseek-ai/deepseek-coder-1.3b-instruct', 'revision': 'e063262dac8366fc1f28a4da0ff3c50ea66259ca'},
}


def metadata():
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    result = {}
    for name, spec in MODELS.items():
        files = {}
        for filename in ('config.json', 'tokenizer_config.json', 'generation_config.json'):
            path = Path(hf_hub_download(spec['id'], filename, revision=spec['revision'], local_files_only=True))
            files[filename] = {'sha256': file_hash(path), 'contents': read_json(path)}
        tokenizer = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'], local_files_only=True,
                                                  trust_remote_code=False)
        result[name] = {**spec, 'tokenizer_revision': spec['revision'], 'files': files,
                        'tokenizer_class': type(tokenizer).__name__, 'bos_id': tokenizer.bos_token_id,
                        'eos_id': tokenizer.eos_token_id, 'pad_id': tokenizer.pad_token_id,
                        'vocabulary': len(tokenizer), 'source': f"https://huggingface.co/{spec['id']}/tree/{spec['revision']}"}
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    hardware = {'measured_at': now(), 'python': platform.python_version(), 'platform': platform.platform(),
                'gpu': props.name if props else None, 'total_memory_mib': props.total_memory / 2**20 if props else 0,
                'cuda_runtime': torch.version.cuda, 'bf16_supported': bool(props and torch.cuda.is_bf16_supported()),
                'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip(),
                'packages': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'numpy', 'scipy',
                            'scikit-learn', 'evalplus', 'huggingface-hub', 'safetensors')}}
    atomic_json(REPORT / 'model_metadata.json', result)
    atomic_json(REPORT / 'hardware.json', hardware)
    return result, hardware


def exposed_ids():
    """Read identities, never collect final labels. Future reservations are not outcomes."""
    ids, provenance = {'Mbpp/2'}, []
    for path in sorted((ROOT / 'runs').rglob('*.jsonl')):
        if 'final_extension' in path.parts:
            continue
        found = set()
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    task = record.get('task_id')
                    if isinstance(task, str) and task.startswith('Mbpp/'):
                        found.add(task)
        if found:
            ids.update(found)
            provenance.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': file_hash(path), 'ids': sorted(found)})
    # Include outcome tables anywhere in reports, including copied historical material.
    for path in sorted((ROOT / 'reports').rglob('*.csv')):
        if 'final_extension' in path.parts:
            continue
        try:
            frame = pd.read_csv(path, usecols=lambda c: c == 'task_id')
        except (ValueError, pd.errors.EmptyDataError):
            continue
        if 'task_id' not in frame:
            continue
        found = {t for t in frame.task_id.dropna().astype(str) if t.startswith('Mbpp/')}
        if found:
            ids.update(found)
            provenance.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': file_hash(path), 'ids': sorted(found)})
    return ids, provenance


def inventory(tasks, benchmark_metadata):
    exposed, provenance = exposed_ids()
    by_id = {t['task_id']: t for t in tasks}
    def normalize(text):
        return re.sub(r'\s+', ' ', re.sub(r'\bdef\s+\w+', 'def FUNCTION', text)).strip().lower()
    normalized = {k: normalize(v['prompt']) for k, v in by_id.items()}
    excluded, pairs = set(exposed), []
    # Conservative lexical near-duplicate screen, fixed before prospective generation.
    ids = sorted(by_id)
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            similarity = SequenceMatcher(None, normalized[left], normalized[right], autojunk=False).ratio()
            if similarity >= .85:
                pairs.append({'left': left, 'right': right, 'ratio': similarity})
                if left in exposed or right in exposed:
                    excluded.update((left, right))
                else:
                    excluded.add(max(left, right))
    eligible = sorted(set(ids) - excluded, key=lambda t: (digest('final-cross-model-v1:' + t), t))
    # Fixed hierarchy if the finite inventory cannot supply 150 independent candidates.
    size = next((n for n in (150, 140, 120, 100, 80) if n <= len(eligible)), 0)
    if not size:
        raise ValueError('Fewer than 80 unexposed eligible tasks; prospectively revise design')
    origin = read_json(ROOT / 'runs/candidate_bank/definitive_v0.2/manifest.json')
    pilot = read_json(ROOT / 'runs/candidate_bank/pilot_v0.2/manifest.json')
    pilot_ids = sorted(pilot['task_ids'], key=lambda t: (digest('final-deepseek-pilot-v1:' + t), t))[:12]
    result = {'created_at': now(), 'dataset': benchmark_metadata, 'total_tasks': len(ids),
              'exposed_ids': sorted(exposed), 'exposure_files': provenance,
              'near_duplicate_rule': 'SequenceMatcher ratio >= .85 after whitespace/function-name normalization',
              'near_duplicate_pairs': pairs, 'excluded_ids': sorted(excluded & set(ids)),
              'eligible_ids': eligible, 'external_ids': eligible[:size], 'external_tasks': size,
              'pilot_ids': pilot_ids, 'source_ids': origin['task_ids'],
              'external_ids_sha256': digest(json.dumps(eligible[:size], separators=(',', ':'))),
              'limits': 'Local exposure inventory and lexical screen cannot certify semantic independence or pretraining non-contamination.'}
    atomic_json(REPORT / 'task_inventory.json', result)
    return result


def historical_audit():
    history_snapshot()
    assert_history()
    bank = ROOT / 'runs/candidate_bank/definitive_v0.2/trajectories.jsonl'
    records = [json.loads(line) for line in bank.read_text(encoding='utf-8').splitlines()]
    by_id = {(r['task_id'], r['attempt_id']): r for r in records}
    if len(by_id) != len(records):
        raise ValueError('Duplicate historical identities')
    p = pd.read_csv(ROOT / 'runs/probes/protocol_v0.2/outer_predictions.csv')
    p = p[p.checkpoint.astype(str) == '32']
    if len(p) != sum(r['generated_tokens'] > 32 for r in records):
        raise ValueError('Historical risk-set denominator differs')
    for r in p.itertuples():
        source = by_id[(r.task_id, r.attempt_id)]
        if int(source['passed']) != r.passed or not (source['generated_tokens'] > 32):
            raise ValueError('Historical prediction/label mismatch')
    for r in records:
        if r['seed'] != attempt_seed('0.2', r['benchmark_version'], r['task_id'], r['attempt_id']):
            raise ValueError('Historical seed mismatch')
        if r['passed'] != all(s == 'pass' for s in r['verifier_status'].values()):
            raise ValueError('Historical label/status mismatch')
    per_task = []
    for task, g in p.groupby('task_id'):
        if g.fold.nunique() != 1:
            raise ValueError('Task crosses historical folds')
        b = task_auc(g.passed, g.B, g.task_id)
        hb = task_auc(g.passed, g.HB, g.task_id)
        if b:
            # Independent pair-count check, with half credit for ties.
            y, scores = g.passed.to_numpy(), g.HB.to_numpy()
            pos, neg = scores[y == 1], scores[y == 0]
            direct = np.mean((pos[:, None] > neg) + .5 * (pos[:, None] == neg))
            if not np.isclose(direct, hb[task]):
                raise ValueError('Independent WT-AUC check failed')
            per_task.append(hb[task] - b[task])
    contrast = paired_task_bootstrap(per_task, replications=2000, seed=20260921)
    expected = read_json(ROOT / 'reports/protocol_v0.2/results.json')['signal']['H2']['delta_hb_minus_b']
    if contrast != expected:
        raise ValueError('Historical contrast not reproduced exactly')
    result = {'completed_at': now(), 'attempts': len(records), 'tasks': len({r['task_id'] for r in records}),
              'active_attempts_t32': len(p), 'mixed_tasks_t32': len(per_task), 'reproduced_contrast': contrast,
              'predictor_correlation': float(p.B.corr(p.HB)), 'label_seed_identity_checks': True,
              'historical_files_unchanged': True,
              'sandbox_finding': 'Historical worker has EvalPlus guards/timeouts but no OS filesystem/network namespace; new evaluations require bubblewrap.',
              'scope': 'Historical labels are checked against recorded base/plus statuses, not rerun in this audit.'}
    atomic_json(REPORT / 'historical_audit.json', result)
    return result
