"""Launch the unchanged frozen pipeline with bounded Windows JSON replace retries."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wsl-python', required=True)
    args = parser.parse_args()
    directory = ROOT / 'reports/research/full_execution_v0.2'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / '.runner.lock').exists():
        raise RuntimeError('Runner lock exists; inspect the existing process before launching another writer')
    support = ROOT / 'scripts/research_runtime'
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(support) + (os.pathsep + environment['PYTHONPATH'] if environment.get('PYTHONPATH') else '')
    command = [sys.executable, '-u', str(ROOT / 'scripts/run_complete_research.py'),
               '--wsl-python', args.wsl_python]
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    recovery = directory / f'io_recovery_{timestamp}'
    recovery.mkdir()
    status_path = directory / 'status.json'
    if status_path.exists():
        raw = status_path.read_bytes()
        (recovery / 'status_before_resume.json').write_bytes(raw)
        status = json.loads(raw)
        status.pop('failure', None)
        status.pop('ended_at', None)
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    paths = [Path(__file__).resolve(), support / 'sitecustomize.py', support / 'slm_windows_atomic_io.py']
    record = {'created_at': datetime.now(timezone.utc).isoformat(), 'supervisor_pid': os.getpid(),
              'command': command, 'runtime_support': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'scope': 'Bounded retries for Windows errors 5/32/33 replacing repository .json.tmp files; no changes to JSON content, numerical computation, seeds, labels or design',
              'frozen_execution_manifest_sha256': hashlib.sha256((ROOT / 'reports/research/execution_freeze_v0.2/manifest.json').read_bytes()).hexdigest()}
    (recovery / 'runtime.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        result = subprocess.run(command, cwd=ROOT, env=environment,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
