"""Run only the isolated, versioned final extension; never rerun historical outputs."""
from pathlib import Path
import argparse
import os
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'report', 'publish', 'design', 'freeze', 'evaluate', 'audit-qwen-labels',
                                          'deepseek-pilot', 'qwen-technical', 'deepseek-source',
                                          'train-qwen', 'train-deepseek', 'qwen-external', 'deepseek-external',
                                          'sensitivity-qwen', 'sensitivity-deepseek'])
    parser.add_argument('--session-seconds', type=int, default=600)
    args = parser.parse_args()
    import torch  # Initialize before scipy/sklearn to avoid Windows OpenMP loader conflicts.
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    from slm_steering.final_extension import workflow
    from slm_steering.final_extension.common import REPORT, ROOT, read_json, file_hash
    if args.command in ('prepare', 'report', 'freeze'):
        getattr(workflow, args.command)()
    elif args.command == 'run':
        from slm_steering.final_extension.common import RUN, atomic_json, now, writer_lock
        with writer_lock(RUN / 'pipeline'):
            try:
                workflow.run_all()
            except BaseException as exc:
                atomic_json(REPORT / 'execution_outcome.json', {'status': 'failed', 'at': now(),
                            'type': type(exc).__name__, 'message': str(exc),
                            'policy': 'No fabricated labels, replacement tasks, or automatic scientific changes'})
                workflow.report()
                raise
    elif args.command == 'design':
        print(workflow.finalize_design(), flush=True)
    elif args.command == 'publish':
        from slm_steering.final_extension.publication import integrate
        workflow.report()
        integrate()
    elif args.command == 'audit-qwen-labels':
        from slm_steering.final_extension.label_audit import run
        run()
        workflow.report()
    elif args.command == 'evaluate':
        from slm_steering.final_extension.evaluation import evaluate
        evaluate()
        workflow.report()
    elif args.command.startswith('sensitivity-'):
        from slm_steering.final_extension.collection import load_features
        from slm_steering.final_extension.training_sensitivity import run
        name = args.command.split('-')[1]
        run(load_features(name), name)
        workflow.report()
    elif args.command.startswith('train-'):
        from slm_steering.final_extension.collection import load_features
        from slm_steering.final_extension.probes import train_all
        name = args.command.split('-')[1]
        train_all(load_features(name), name)
        workflow.report()
    else:
        from slm_steering.final_extension.collection import collect
        name, phase = args.command.split('-')
        if phase in ('pilot', 'technical'):
            plan = read_json(REPORT / 'pilot_plan.json')
            for path, expected in plan['files'].items():
                if file_hash(ROOT / path) != expected:
                    raise ValueError(f'Pilot source changed since prospective lock: {path}')
        try:
            collect(name, phase, session_seconds=args.session_seconds)
        finally:
            workflow.report()


if __name__ == '__main__':
    main()
