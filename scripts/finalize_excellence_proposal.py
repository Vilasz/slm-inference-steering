"""Finalize reviewable resource estimates and proposed hashes, without granting approval."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from slm_steering.excellence.approvals import approval_hashes
from slm_steering.excellence.audit import assert_preserved
from slm_steering.research.protocol import file_hash, now, read_bank, write_json


def main():
    if (ROOT / 'docs/protocols/EXCELLENCE_EXTENSION_v1.approval.json').exists():
        raise ValueError('A reviewed proposal must not be overwritten; prepare an explicit amendment')
    assert_preserved(ROOT)
    directory = ROOT / 'reports/excellence'
    hardware = json.loads((directory / 'hardware/results.json').read_text(encoding='utf-8'))
    timing = json.loads((directory / 'hardware/component_timings.json').read_text(encoding='utf-8'))
    pairs = [json.loads(line) for line in (directory / 'hardware/pairs.jsonl').read_text(encoding='utf-8').splitlines()]
    prefill = float(np.mean([r['timings']['selective']['prefill_gpu_seconds'] for r in pairs]))
    tokens = sum(r['tokens'] for r in pairs)
    seconds = sum(r['timings']['selective']['generation_seconds'] for r in pairs)
    rest_per_token = (seconds - prefill * len(pairs)) / tokens
    records = read_bank(ROOT / 'runs/candidate_bank/transfer_v0.2/trajectories.jsonl')
    target_length = float(np.mean([r['generated_tokens'] for r in records]))
    per_attempt = prefill + rest_per_token * target_length + timing['mean_verifier_rpc_seconds']
    scenarios = [{'main_tasks': tasks, 'technical_tasks': 20, 'attempts': (tasks + 20) * 20,
                  'expected_hours': (tasks + 20) * 20 * per_attempt / 3600,
                  'hours_with_50pct_buffer': 1.5 * (tasks + 20) * 20 * per_attempt / 3600} for tasks in (100, 120)]
    budget = {'created_at': now(), 'hardware': hardware['hardware'], 'known_candidate_pairs': len(pairs),
              'generation_seconds_per_candidate_reference': hardware['mean_selective_seconds'],
              'reference_mean_tokens': hardware['mean_tokens'], 'reference_prefill_event_span_seconds': prefill,
              'nonprefill_wall_seconds_per_token': rest_per_token, 'historical_target_mean_length': target_length,
              'forecast_seconds_per_attempt_including_verification': per_attempt, 'scenarios': scenarios,
              'limitations': ['Linear length extrapolation from known HumanEval pilot to new MBPP tasks is a planning approximation',
                             '50% buffer covers variability and interruptions but is not a probabilistic bound',
                             'CUDA event spans include host scheduling and capture gaps; not isolated kernel FLOPs',
                             'No extrapolation to an unavailable RTX4060 and no policy wall-clock savings claimed']}
    write_json(directory / 'resource_budget.json', budget)
    design_path = directory / 'proposal_design.json'
    design = json.loads(design_path.read_text(encoding='utf-8'))
    dataset = json.loads((directory / 'mbpp_available_task_ids.json').read_text(encoding='utf-8'))
    design.update(approved_hardware_device=hardware['hardware']['device'], benchmark_sha256=dataset['metadata']['dataset_sha256'],
                  bundle_sha256=file_hash(ROOT / 'runs/transfer/protocol_v0.2/models.joblib'),
                  policies={'B': {'checkpoint': 8, 'threshold': .3166734153299228},
                            'HB': {'checkpoint': 8, 'threshold': .26953905544150975}},
                  bootstrap_replications=10000, bootstrap_seed=20261017,
                  primary_estimand='Candidate-count recall HB minus B on shared active-correct t8 population',
                  success_rule='Upper endpoint of two-sided task-bootstrap 95% CI below zero',
                  resource_budget_sha256=file_hash(directory / 'resource_budget.json'))
    write_json(design_path, design)
    record = {'created_at': now(), 'status': 'unapproved_proposal_hashes_only',
              'student_approval': 'pending', 'advisor_review': 'pending', **approval_hashes(ROOT)}
    write_json(directory / 'proposed_freeze_manifest.json', record)
    lines = ['# Orçamento e revisão da extensão independente', '',
             f"Dispositivo: {hardware['hardware']['device']}; {hardware['hardware']['memory_total_mib']:.0f} MiB.",
             '50 pares / 100 gerações; todos os tokens e estados correspondentes coincidiram exatamente.',
             f"Captura original: {hardware['mean_original_seconds']:.3f} s/candidato; seletiva: {hardware['mean_selective_seconds']:.3f} s/candidato (descritivo).",
             f"Prefill, span de eventos CUDA: {prefill:.4f} s; verificação RPC média: {timing['mean_verifier_rpc_seconds']:.4f} s.", '',
             '| Tarefas principais + técnicas | Tentativas | Horas esperadas | Com margem de 50% |', '|---|---:|---:|---:|']
    for scenario in scenarios:
        lines.append(f"| {scenario['main_tasks']} + 20 | {scenario['attempts']} | {scenario['expected_hours']:.2f} | {scenario['hours_with_50pct_buffer']:.2f} |")
    lines += ['', 'Planejamento condicionado aos tempos e comprimentos observados, sem garantia da duração futura.',
              'A nova coleta permanece pendente de aprovação do estudante e revisão do orientador.',
              'Protocolo: `docs/protocols/EXCELLENCE_EXTENSION_v1.md`.',
              'Hashes propostos: `proposed_freeze_manifest.json`; a existência do arquivo não constitui congelamento ou aprovação.', '',
              'A comparação principal usará B e H+B em 8 tokens, sem ajuste de limiar no alvo.',
              'As alternativas de 100 ou 120 tarefas devem ser decididas antes dos resultados novos.', '']
    (directory / 'resource_budget.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'status': record['status'], 'budget': scenarios, 'protocol_sha256': record['protocol_sha256']}, indent=2))


if __name__ == '__main__':
    main()
