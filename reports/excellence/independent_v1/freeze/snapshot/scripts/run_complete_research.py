"""Sequential, resumable full local experiment. Stops on failed scientific gates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from slm_steering.research.freeze import validate_execution_freeze
from slm_steering.research.protocol import file_hash, now, write_json


def process_alive(pid: int) -> bool:
    if os.name == 'nt':
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wsl-python', required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/protocol_v0.2.yaml')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    pilot = ROOT / 'runs/candidate_bank/pilot_v0.2'
    main_bank = ROOT / 'runs/candidate_bank/definitive_v0.2'
    target = ROOT / 'runs/candidate_bank/transfer_v0.2'
    pilot_manifest = pilot / 'manifest.json'
    budget = ['--time-budget-seconds', '1000000']
    generation = ['--config', str(args.config), '--wsl-python', args.wsl_python, *budget]
    probes = ROOT / 'runs/probes/protocol_v0.2'
    stages = [
        ('pilot', 'build_candidate_bank.py', ['--phase', 'pilot', '--output-dir', str(pilot), *generation], pilot_manifest),
        ('pilot_audit', 'audit_candidate_bank.py', ['--bank-dir', str(pilot), '--output', 'reports/research/pilot_v0.2_audit.json'], ROOT / 'reports/research/pilot_v0.2_audit.json'),
        ('pilot_prefix_validation', 'validate_bank_prefixes.py', ['--bank-dir', str(pilot), '--config', str(args.config)], pilot / 'prefix_validation.json'),
        ('sampling_freeze', 'prepare_sampling_freeze.py', ['--pilot-dir', str(pilot), '--config', str(args.config)], ROOT / 'configs/sampling_freeze_v0.2.json'),
        ('main_bank', 'build_candidate_bank.py', ['--phase', 'definitive', '--output-dir', str(main_bank), '--pilot-manifest', str(pilot_manifest), *generation], main_bank / 'manifest.json'),
        ('main_audit', 'audit_candidate_bank.py', ['--bank-dir', str(main_bank), '--output', 'reports/research/definitive_v0.2_audit.json'], ROOT / 'reports/research/definitive_v0.2_audit.json'),
        ('main_prefix_validation', 'validate_bank_prefixes.py', ['--bank-dir', str(main_bank), '--config', str(args.config)], main_bank / 'prefix_validation.json'),
        ('grouped_probes', 'run_grouped_probes.py', ['--bank-dir', str(main_bank), '--config', str(args.config)], probes / 'manifest.json'),
        ('online_validation', 'validate_online_replay.py', ['--bank-dir', str(main_bank), '--config', str(args.config), '--wsl-python', args.wsl_python, '--max-tasks', '12'], probes / 'online_validation.json'),
        ('main_results', 'generate_tcc_artifacts.py', ['--config', str(args.config)], ROOT / 'reports/protocol_v0.2/results.json'),
        ('transfer_freeze', 'run_frozen_transfer.py', ['prepare', '--bank-dir', str(main_bank), '--config', str(args.config)], ROOT / 'runs/transfer/protocol_v0.2/models.joblib'),
        ('transfer_bank', 'build_candidate_bank.py', ['--phase', 'transfer', '--output-dir', str(target), '--pilot-manifest', str(pilot_manifest), '--transfer-bundle', 'runs/transfer/protocol_v0.2/models.joblib', *generation], target / 'manifest.json'),
        ('transfer_audit', 'audit_candidate_bank.py', ['--bank-dir', str(target), '--output', 'reports/research/transfer_v0.2_audit.json'], ROOT / 'reports/research/transfer_v0.2_audit.json'),
        ('transfer_prefix_validation', 'validate_bank_prefixes.py', ['--bank-dir', str(target), '--config', str(args.config)], target / 'prefix_validation.json'),
        ('transfer_results', 'run_frozen_transfer.py', ['evaluate', '--bank-dir', str(target), '--config', str(args.config)], ROOT / 'reports/transfer_v0.2/results.json'),
        ('delivery', 'build_research_delivery.py', ['--as-of', now()[:10]], ROOT / 'reports/final_review/evidence.json'),
        ('notebook', 'build_final_notebook.py', ['--execute'], ROOT / 'notebooks/final_research_summary.ipynb'),
        ('manuscript', 'build_manuscript.py', ['--compile'], ROOT / 'tcc/build/manuscrito.pdf'),
        ('publication_audit', 'audit_research_delivery.py', [], ROOT / 'reports/final_review/delivery_manifest.json'),
    ]
    if args.dry_run:
        print(json.dumps([{'stage': name, 'script': script, 'args': params} for name, script, params, _ in stages], indent=2))
        return
    validate_execution_freeze(ROOT, args.config.resolve())
    directory = ROOT / 'reports/research/full_execution_v0.2'
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / '.runner.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    status_path = directory / 'status.json'
    status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {'started_at': now(), 'stages': {}}
    status.update(status='running', pid=os.getpid(), config_sha256=file_hash(args.config))
    write_json(status_path, status)
    try:
        for name, script, params, output in stages:
            previous = status['stages'].get(name, {})
            if previous.get('status') == 'completed':
                if not output.exists() or file_hash(output) != previous['output_sha256']:
                    raise ValueError(f'Previously completed stage changed: {name}')
                continue
            status.update(current_stage=name, updated_at=now())
            status['stages'][name] = {'status': 'running', 'started_at': now()}
            write_json(status_path, status)
            if name == 'delivery':
                params = ['--as-of', now()[:10]]
            if name == 'pilot':
                writer = pilot / '.writer.lock'
                while writer.exists():
                    pid = int(writer.read_text(encoding='utf-8'))
                    if not process_alive(pid):
                        raise RuntimeError(f'Stale pilot writer lock for PID {pid}; inspect the bank before recovery')
                    print(f'{now()} Waiting for existing pilot writer PID {pid}', flush=True)
                    time.sleep(20)
            command = [sys.executable, '-u', str(ROOT / 'scripts' / script), *params]
            print(f'{now()} Stage: {name}', flush=True)
            with (directory / f'{name}.log').open('a', encoding='utf-8') as log:
                log.write(f'\n{now()} {json.dumps(command)}\n')
                log.flush()
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Stage {name} exited {result.returncode}; see {directory / (name + ".log")}')
            if name in {'pilot', 'main_bank', 'transfer_bank', 'grouped_probes'}:
                manifest = json.loads(output.read_text(encoding='utf-8'))
                if manifest['status'] != 'completed':
                    raise ValueError(f'Stage {name} did not complete its planned sample')
            if not output.exists():
                raise ValueError(f'Stage {name} did not produce its declared output')
            status['stages'][name].update(status='completed', ended_at=now(), output_sha256=file_hash(output))
            write_json(status_path, status)
        status.update(status='completed', ended_at=now())
        write_json(status_path, status)
    except BaseException as exc:
        status.update(status='failed', ended_at=now(), failure={'type': type(exc).__name__, 'message': str(exc)})
        write_json(status_path, status)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
