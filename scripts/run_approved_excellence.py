"""Supervise the approved extension in durable sessions; stop on technical errors."""
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
import time

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'reports/excellence/independent_v1/execution'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/research_runtime'))


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def process_alive(pid):
    if os.name == 'nt':
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        code = ctypes.c_ulong()
        try:
            return bool(ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def phase_manifest(phase):
    path = ROOT / f'runs/excellence/independent_v1/{phase}/manifest.json'
    return read(path) if path.exists() else {}


def verify_freeze():
    from slm_steering.excellence.approvals import approved_design
    approved_design(ROOT)
    freeze = read(ROOT / 'reports/excellence/independent_v1/freeze/manifest.json')
    for relative, expected in freeze['inputs'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Approved frozen input changed: {relative}')


def run_child(command, state, label):
    path = DIRECTORY / f'{label}.log'
    with path.open('a', encoding='utf-8') as log:
        log.write(f'\n{stamp()} command={json.dumps(command)}\n')
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        state.update(child_pid=process.pid, current_log=path.relative_to(ROOT).as_posix())
        while process.poll() is None:
            state.update(updated_at=stamp(), banks={p: {k: v for k, v in phase_manifest(p).items()
                if k in ('status', 'completed_attempts', 'expected_attempts', 'numerical_checks_passed')}
                for p in ('technical', 'main')})
            write(DIRECTORY / 'status.json', state)
            time.sleep(5)
        state['child_pid'] = None
        if process.returncode:
            raise RuntimeError(f'{label} failed with exit code {process.returncode}; inspect {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-pid', type=int)
    parser.add_argument('--wsl-python', default='/home/joaof/.venvs/slm-evalplus-v031/bin/python')
    args = parser.parse_args()
    from slm_windows_atomic_io import install
    install()
    verify_freeze()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    lock = DIRECTORY / '.supervisor.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    state = {'started_at': stamp(), 'supervisor_pid': os.getpid(), 'status': 'running',
             'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'initial_collector_pid': args.wait_for_pid, 'sessions': [], 'phase': 'technical'}
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        if args.wait_for_pid:
            while process_alive(args.wait_for_pid):
                state.update(updated_at=stamp(), phase='technical_existing_session',
                             banks={'technical': {k: v for k, v in phase_manifest('technical').items()
                                if k in ('status', 'completed_attempts', 'expected_attempts', 'numerical_checks_passed')}})
                write(DIRECTORY / 'status.json', state)
                time.sleep(5)
            if (ROOT / 'runs/excellence/independent_v1/technical/failure.json').exists():
                raise RuntimeError('Initial technical collector reported a failure; review before resuming')
        for phase in ('technical', 'main'):
            state['phase'] = phase
            while phase_manifest(phase).get('status') != 'completed':
                verify_freeze()
                if (ROOT / f'runs/excellence/independent_v1/{phase}/.writer.lock').exists():
                    raise RuntimeError('Another collector holds the writer lock; do not start a second writer')
                previous = phase_manifest(phase).get('completed_attempts', 0)
                label = f'{phase}_{previous:04d}_{datetime.now(timezone.utc).strftime("%H%M%S")}'
                run_child([sys.executable, '-u', str(ROOT / 'scripts/run_excellence_extension.py'),
                           '--phase', phase, '--time-budget-seconds', '600', '--wsl-python', args.wsl_python], state, label)
                manifest = phase_manifest(phase)
                if manifest.get('completed_attempts', 0) <= previous:
                    raise RuntimeError('Collector made no progress; do not spin on a failed session')
                state['sessions'].append({'phase': phase, 'completed_attempts': manifest['completed_attempts'], 'ended_at': stamp()})
        verify_freeze()
        state['phase'] = 'evaluation'
        run_child([sys.executable, '-u', str(ROOT / 'scripts/run_excellence_extension.py'), '--evaluate'], state, 'evaluation')
        state.update(status='completed_collection_and_primary_evaluation', phase='publication_pending', ended_at=stamp())
        write(DIRECTORY / 'status.json', state)
        publisher = ROOT / 'scripts/publish_excellence_independent.py'
        if publisher.exists():
            state.update(status='running', phase='publication')
            run_child([sys.executable, '-u', str(publisher)], state, 'publication')
            state.update(status='completed', phase='completed', ended_at=stamp())
    except BaseException as exc:
        state.update(status='failed', failed_at=stamp(), failure={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        state.update(updated_at=stamp(), banks={p: {k: v for k, v in phase_manifest(p).items()
            if k in ('status', 'completed_attempts', 'expected_attempts', 'numerical_checks_passed')}
            for p in ('technical', 'main')})
        write(DIRECTORY / 'status.json', state)
        lock.unlink(missing_ok=True)
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()
