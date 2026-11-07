from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

from slm_steering.research.protocol import ROOT, canonical, file_hash, now
from slm_steering.research.protocol import digest as digest

REPORT = ROOT / 'reports/final_extension'
RUN = ROOT / 'runs/final_extension/v1'
PROTOCOL = ROOT / 'docs/protocols/FINAL_CROSS_MODEL_EXTENSION_v1.md'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != json.loads(json.dumps(value)):
            raise ValueError(f'Immutable artifact differs: {path}')
    else:
        atomic_json(path, value)


@contextmanager
def writer_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / '.writer.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(canonical({'pid': os.getpid(), 'started_at': now()}))
    try:
        yield
    finally:
        lock.unlink()


def history_snapshot():
    path = REPORT / 'historical_hashes.json'
    if path.exists():
        return read_json(path)
    baseline = read_json(ROOT / 'reports/excellence/baseline.json')
    paths = set(baseline['protected'])
    for folder in ('runs/excellence/independent_v1', 'reports/excellence/independent_v1'):
        paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT / folder).rglob('*')
                     if p.is_file() and p.suffix in {'.json', '.jsonl', '.csv', '.md'})
    paths.update(['reports/excellence/models/results.json', 'reports/excellence/economics/results.json'])
    result = {'captured_at': now(), 'files': {p: file_hash(ROOT / p) for p in sorted(paths) if (ROOT / p).exists()}}
    atomic_json(path, result)
    return result


def assert_history():
    for path, expected in history_snapshot()['files'].items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f'Historical artifact changed: {path}')


def validate_freeze():
    freeze = read_json(REPORT / 'freeze.json')
    for path, expected in freeze['files'].items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f'Frozen source changed; prospectively amend before use: {path}')
    assert_history()
    return freeze
