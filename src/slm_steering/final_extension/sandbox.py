"""EvalPlus inside an unprivileged WSL namespace, with no host home or network."""
from __future__ import annotations

import json
import os
import subprocess

from slm_steering.final_extension.common import ROOT
from slm_steering.research.evaluation import wsl_path

BWRAP = '/home/joaof/.local/slm-sandbox/usr/bin/bwrap'
VENV = '/home/joaof/.venvs/slm-evalplus-v031'
CACHE = '/home/joaof/.cache/evalplus'


def sandbox_command(benchmark, *, probe=False):
    worker = wsl_path(ROOT / 'scripts/final_extension_worker.py')
    original = wsl_path(ROOT / 'scripts/evalplus_worker.py')
    command = [BWRAP, '--unshare-all', '--die-with-parent', '--new-session', '--cap-drop', 'ALL',
               '--clearenv', '--setenv', 'PATH', '/opt/venv/bin:/usr/bin',
               '--setenv', 'HOME', '/tmp', '--setenv', 'XDG_CACHE_HOME', '/cache',
               '--setenv', 'OPENBLAS_NUM_THREADS', '1', '--setenv', 'OMP_NUM_THREADS', '1',
               '--setenv', 'EVALPLUS_MAX_MEMORY_BYTES', '3221225472',
               '--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib', '--ro-bind', '/lib64', '/lib64',
               '--ro-bind', VENV, '/opt/venv', '--ro-bind', CACHE, '/cache/evalplus',
               '--ro-bind', worker, '/app/worker.py', '--ro-bind', original, '/app/evalplus_worker.py',
               '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--chdir', '/tmp', '--remount-ro', '/',
               '/usr/bin/prlimit', '--as=3221225472', '--cpu=7200', '--nproc=64', '--fsize=33554432',
               '/opt/venv/bin/python', '-I', '-u', '/app/worker.py', '--benchmark', benchmark]
    if probe:
        command += ['--probe']
    return ['wsl', '-d', 'Ubuntu', '--', *command] if os.name == 'nt' else command


class SandboxedEvalPlus:
    def __init__(self, benchmark):
        self.process = subprocess.Popen(sandbox_command(benchmark), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, encoding='utf-8', bufsize=1)
        try:
            self.dataset = self.request({'operation': 'dataset'})
        except BaseException:
            self.close()
            raise

    def request(self, value):
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        self.process.stdin.write(json.dumps(value) + '\n')
        self.process.stdin.flush()
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(self.process.stdout.readline)
        try:
            line = future.result(timeout=300)
        except TimeoutError:
            self.process.kill()
            raise RuntimeError('Sandbox RPC timed out; no label may be fabricated')
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        if not line:
            raise RuntimeError(f'Sandbox exited: {self.process.poll()}')
        response = json.loads(line)
        if 'error' in response:
            raise RuntimeError(response['error'])
        return response['result']

    def verify(self, task_id, code):
        return self.request({'operation': 'verify', 'task_id': task_id, 'code': code})

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"operation":"close"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait(timeout=10)
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def probe_sandbox():
    result = subprocess.run(sandbox_command('humaneval', probe=True), capture_output=True, text=True,
                            encoding='utf-8', timeout=45, check=True)
    value = json.loads(result.stdout)
    if not all(value.values()):
        raise RuntimeError(f'Sandbox restrictions failed: {value}')
    return value
