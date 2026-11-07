from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from slm_steering.research.protocol import ROOT


def wsl_path(path: Path) -> str:
    path = path.resolve()
    return '/mnt/' + path.drive[0].lower() + path.as_posix()[2:]


class EvalPlusClient:
    def __init__(self, benchmark: str, *, wsl_python: str | None = None, distro: str = 'Ubuntu'):
        worker = ROOT / 'scripts' / 'evalplus_worker.py'
        if os.name == 'nt':
            if not wsl_python:
                raise ValueError('Provide --wsl-python for official EvalPlus on Windows')
            command = ['wsl', '-d', distro, '--', wsl_python, '-u', wsl_path(worker), '--benchmark', benchmark]
        else:
            command = [sys.executable, '-u', str(worker), '--benchmark', benchmark]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, encoding='utf-8', bufsize=1)
        self.dataset = self.request({'operation': 'dataset'})

    def request(self, value: dict) -> dict:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(value) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'EvalPlus worker exited ({self.process.poll()}); labels cannot be substituted')
        response = json.loads(line)
        if 'error' in response:
            raise RuntimeError(response['error'])
        return response['result']

    def verify(self, task_id: str, code: str) -> dict:
        return self.request({'operation': 'verify', 'task_id': task_id, 'code': code})

    def close(self):
        if self.process.poll() is None:
            assert self.process.stdin is not None
            self.process.stdin.write('{"operation":"close"}\n')
            self.process.stdin.flush()
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
