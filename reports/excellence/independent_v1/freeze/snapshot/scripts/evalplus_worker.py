"""Small POSIX-only RPC worker using EvalPlus's official oracle and checker.

Install EvalPlus 0.3.1 and its evaluation-only dependencies in this environment.
No CUDA, model weights, torch or language-model API clients are needed here.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import platform
import sys
import time


def dataset_json_default(value):
    """Canonical typed encoding of non-JSON benchmark inputs for provenance.

    Does not modify the actual objects passed to EvalPlus. HumanEval's JSON
    representation is unchanged; MBPP deserializes some inputs as complex values.
    """
    if isinstance(value, complex):
        return {'__evalplus_type__': 'complex', 'real': value.real, 'imag': value.imag}
    if isinstance(value, (set, frozenset)):
        elements = sorted(value, key=lambda item: json.dumps(item, sort_keys=True, default=dataset_json_default))
        return {'__evalplus_type__': type(value).__name__, 'items': elements}
    raise TypeError(f'Unsupported benchmark input type: {type(value).__name__}')


def serve(benchmark: str) -> None:
    if sys.platform == 'win32':
        raise RuntimeError('EvalPlus signal timeouts require POSIX; use WSL')
    from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash, get_mbpp_plus, get_mbpp_plus_hash
    from evalplus.data.humaneval import HUMANEVAL_PLUS_VERSION
    from evalplus.data.mbpp import MBPP_PLUS_VERSION
    from evalplus.eval import PASS, untrusted_check
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.gen.util import trusted_exec

    with contextlib.redirect_stdout(sys.stderr):
        problems = get_human_eval_plus() if benchmark == 'humaneval' else get_mbpp_plus()
        dataset_hash = get_human_eval_plus_hash() if benchmark == 'humaneval' else get_mbpp_plus_hash()
    oracles = {}
    metadata = {'benchmark': benchmark, 'version': HUMANEVAL_PLUS_VERSION if benchmark == 'humaneval' else MBPP_PLUS_VERSION,
                'evalplus_version': importlib.metadata.version('evalplus'), 'official_dataset_md5': dataset_hash,
                'dataset_sha256': hashlib.sha256(json.dumps(problems, sort_keys=True,
                    default=dataset_json_default).encode()).hexdigest(),
                'python': platform.python_version(), 'platform': platform.platform(),
                'checker': 'evalplus.eval.untrusted_check', 'fast_check': True, 'mini': False, 'noextreme': False,
                'minimum_time_limit': 1, 'groundtruth_time_factor': 4}
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request['operation'] == 'dataset':
                result = {'metadata': metadata, 'tasks': [{'task_id': key, 'prompt': p['prompt'],
                         'entry_point': p['entry_point']} for key, p in problems.items()]}
            elif request['operation'] == 'verify':
                task_id = request['task_id']
                problem = problems[task_id]
                if task_id not in oracles:
                    canonical_code = problem['prompt'] + problem['canonical_solution']
                    special = benchmark == 'mbpp' and problem['entry_point'] in MBPP_OUTPUT_NOT_NONE_TASKS
                    oracles[task_id] = {kind: trusted_exec(canonical_code, problem[kind + '_input'],
                        problem['entry_point'], record_time=True, output_not_none=special) for kind in ('base', 'plus')}
                code = request.get('code')
                if request.get('canonical_control'):
                    code = problem['prompt'] + problem['canonical_solution']
                start = time.perf_counter()
                statuses = {}
                for kind in ('base', 'plus'):
                    expected, times = oracles[task_id][kind]
                    status, _details = untrusted_check(benchmark, code, problem[kind + '_input'],
                        problem['entry_point'], expected, problem['atol'], times, fast_check=True,
                        min_time_limit=1, gt_time_limit_factor=4)
                    statuses[kind] = status
                error = None
                if any(value != PASS for value in statuses.values()):
                    try:
                        compile(code, '<candidate>', 'exec')
                    except SyntaxError:
                        error = 'syntax_error'
                    else:
                        # EvalPlus intentionally hides exception details. Do not invent
                        # a runtime/assertion distinction not returned by its checker.
                        error = 'timeout' if 'timeout' in statuses.values() else 'evalplus_fail'
                result = {'passed': error is None, 'error_type': error, 'verifier_status': statuses,
                          'verification_time_seconds': time.perf_counter() - start,
                          'test_counts': {kind: len(problem[kind + '_input']) for kind in ('base', 'plus')}}
            elif request['operation'] == 'close':
                break
            else:
                raise ValueError('Unknown worker operation')
            print(json.dumps({'result': result}, allow_nan=False), flush=True)
        except Exception as exc:
            print(json.dumps({'error': f'{type(exc).__name__}: {exc}'}), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=['humaneval', 'mbpp'], default='humaneval')
    serve(parser.parse_args().benchmark)
