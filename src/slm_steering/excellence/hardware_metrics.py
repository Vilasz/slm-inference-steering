"""Separate verifier RPC, CPU gate overhead, and a conservative local resource forecast."""
from __future__ import annotations

import json
from pathlib import Path
import time

import joblib
import numpy as np

from slm_steering.excellence.audit import assert_preserved, checkpoint_data
from slm_steering.research.evaluation import EvalPlusClient
from slm_steering.research.features import observable_features
from slm_steering.research.probes import calibrated_scores
from slm_steering.research.protocol import now, read_bank, write_json


def run(root: Path):
    assert_preserved(root)
    output = root / 'reports/excellence/hardware'
    plan = json.loads((output / 'plan.json').read_text(encoding='utf-8'))
    bank = read_bank(root / 'runs/candidate_bank/pilot_v0.2/trajectories.jsonl')
    lookup = {(r['task_id'], r['attempt_id']): r for r in bank}
    verifications = []
    with EvalPlusClient('humaneval', wsl_python='/home/joaof/.venvs/slm-evalplus-v031/bin/python') as client:
        for identity in plan['identities']:
            r = lookup[(identity['task_id'], identity['attempt_id'])]
            start = time.perf_counter()
            result = client.verify(r['task_id'], r['extracted_code'])
            wall = time.perf_counter() - start
            if result['passed'] != r['passed']:
                raise ValueError('Known-candidate verifier label drift; do not rewrite the historical label')
            verifications.append({'task_id': r['task_id'], 'attempt_id': r['attempt_id'],
                                  'checker_seconds': result['verification_time_seconds'], 'rpc_wall_seconds': wall,
                                  'historical_label_reproduced': True})
    bundle = joblib.load(root / 'runs/transfer/protocol_v0.2/models.joblib')
    data = checkpoint_data(root, checkpoint=8)
    timings = []
    for family in ('B', 'HB'):
        fitted = bundle['models'][(family, 8)]
        def predict(index):
            obs = observable_features(data['records'][index], 8)
            x = obs if family == 'B' else np.r_[data['hidden'][index, data['layers'].index(fitted['layer'])], obs]
            return calibrated_scores(fitted, x.reshape(1, -1))[0]
        for _ in range(10):
            predict(0)
        values = []
        for index in range(100):
            start = time.perf_counter()
            p = predict(index % 20)
            values.append(time.perf_counter() - start)
            if not np.isfinite(p):
                raise ValueError('Gate score is not finite')
        timings.append({'family': family, 'repetitions': 100, 'mean_seconds': float(np.mean(values)),
                        'p50_seconds': float(np.median(values)), 'p95_seconds': float(np.quantile(values, .95)),
                        'scope': 'Warm CPU observable construction and scaler/logistic/Platt inference; excludes disk IO and GPU-to-CPU transfer'})
    result = {'created_at': now(), 'status': 'completed', 'known_candidate_verifications': verifications,
              'mean_checker_seconds': float(np.mean([r['checker_seconds'] for r in verifications])),
              'mean_verifier_rpc_seconds': float(np.mean([r['rpc_wall_seconds'] for r in verifications])), 'gate_microbenchmarks': timings,
              'limitations': ['Warm microbenchmarks are not end-to-end policy latency',
                             'Capture benchmark was partly concurrent with CPU exploratory analysis; timings are descriptive',
                             'Verifier includes first-task oracle setup in RPC wall time; checker-internal time is reported separately']}
    write_json(output / 'component_timings.json', result)
    print('Component timings completed: 50 known labels reproduced; CPU gates warmed and measured', flush=True)
    assert_preserved(root)
