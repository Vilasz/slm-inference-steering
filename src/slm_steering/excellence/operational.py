"""Bounded descriptive end-to-end timing on already examined pilot tasks.

This module does not collect new scientific tasks or fit any predictor.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import time

import joblib
import numpy as np
import pandas as pd

from slm_steering.excellence.audit import assert_preserved
from slm_steering.research.features import observable_features
from slm_steering.research.generation import PrefixGenerator, sampling_statistics
from slm_steering.research.probes import calibrated_scores
from slm_steering.research.protocol import canonical, digest, file_hash, load_config, now, read_bank, write_json


MODES = ('verifier_plain', 'verifier_selective', 'B', 'HB')


def generate(engine, record, mode, bundle, policies):
    """Generate with no capture, one selective capture, or a causal t=8 gate."""
    from transformers.cache_utils import DynamicCache
    from slm_steering.activations.hooks import resolve_transformer_layers
    from slm_steering.generation import render_code_prompt
    torch = engine.torch
    seed = record['seed']
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = torch.tensor([engine.tokenizer.encode(render_code_prompt(engine.tokenizer, record['prompt']))], device=engine.device)
    captured, handles = {}, []
    capture = False
    copy_seconds = score_seconds = 0.
    if mode in ('HB', 'verifier_selective'):
        def hook(_module, _inputs, output):
            nonlocal copy_seconds
            if capture:
                start = time.perf_counter()
                hidden = output[0] if isinstance(output, tuple) else output
                captured['8'] = hidden[0, -1].detach().float().cpu().numpy().astype(np.float16)
                copy_seconds += time.perf_counter() - start
        layer = bundle['models'][('HB', 8)]['layer']
        handles.append(resolve_transformer_layers(engine.model)[layer].register_forward_hook(hook))
    tokens, logprobs, entropies, margins = [], [], [], []
    aborted, probability = False, None
    engine._sync()
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            output = engine.model(input_ids=inputs, past_key_values=DynamicCache(), use_cache=True)
            engine._sync()
            prefill_seconds = time.perf_counter() - start
            for offset in range(1, engine.config['decoding']['max_new_tokens'] + 1):
                probabilities, lp, entropy, margin = sampling_statistics(output.logits[:, -1], 1., engine.config['decoding']['top_p'])
                token = torch.multinomial(probabilities, num_samples=1)
                token_id = int(token.item())
                tokens.append(token_id)
                logprobs.append(float(lp[0, token_id].item()))
                entropies.append(float(entropy.item()))
                margins.append(float(margin.item()))
                terminal = token_id in engine.eos_ids or offset == engine.config['decoding']['max_new_tokens']
                capture = offset == 8 and not terminal
                output = engine.model(input_ids=token, past_key_values=output.past_key_values, use_cache=True)
                if terminal:
                    break
                if offset == 8 and mode in ('B', 'HB'):
                    score_start = time.perf_counter()
                    prefix = {**record, 'generated_tokens': 8, 'token_logprobs': logprobs,
                              'token_entropies': entropies, 'token_margins': margins}
                    obs = observable_features(prefix, 8)
                    x = obs if mode == 'B' else np.r_[captured['8'].astype(float), obs]
                    probability = float(calibrated_scores(bundle['models'][(mode, 8)], x[None, :])[0])
                    score_seconds += time.perf_counter() - score_start
                    if probability < policies[mode]['threshold']:
                        aborted = True
                        break
            engine._sync()
    finally:
        for handle in handles:
            handle.remove()
    return {'token_ids': tokens, 'raw_completion': engine.tokenizer.decode(tokens, skip_special_tokens=True),
            'captured': captured, 'aborted': aborted, 'probability': probability,
            'generation_seconds': time.perf_counter() - start, 'prefill_seconds': prefill_seconds,
            'capture_seconds': copy_seconds, 'classification_seconds': score_seconds}


def run(root: Path, *, maximum_seconds=1800):
    assert_preserved(root)
    main = root / 'runs/excellence/independent_v1/main'
    manifest = json.loads((main / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'completed' or (main / '.writer.lock').exists():
        raise RuntimeError('Independent collection must finish before the GPU timing experiment')
    output = root / 'reports/excellence/operational_v1'
    output.mkdir(exist_ok=True)
    source_hash = file_hash(Path(__file__))
    if (output / 'results.json').exists():
        previous = json.loads((output / 'results.json').read_text(encoding='utf-8'))
        if previous['source_sha256'] != source_hash:
            raise ValueError('Preserve completed timing source; use a new version for a change')
        return previous
    records = [r for r in read_bank(root / 'runs/candidate_bank/pilot_v0.2/trajectories.jsonl')
               if r['decoding_config']['temperature'] == 1.]
    grouped = {task: sorted([r for r in records if r['task_id'] == task], key=lambda r: r['attempt_id'])[:3]
               for task in sorted({r['task_id'] for r in records})}
    tasks = sorted(grouped, key=lambda task: (np.median([r['generated_tokens'] for r in grouped[task]]), task))
    selected = []
    for stratum in np.array_split(np.arange(len(tasks)), 3):
        selected += sorted([tasks[i] for i in stratum], key=lambda task: digest('excellence-operational-v1:' + task))[:2]
    config = load_config(root / 'configs/protocol_v0.2.yaml')
    bundle = joblib.load(root / 'runs/transfer/protocol_v0.2/models.joblib')
    design = json.loads((root / 'reports/excellence/proposal_design.json').read_text(encoding='utf-8'))
    plan = {'created_at': now(), 'source_sha256': source_hash, 'task_ids': selected,
            'selection': 'Two SHA256-selected tasks in each of three pilot median-length strata, no correctness filtering',
            'modes': list(MODES), 'repetitions': 3, 'maximum_attempts': 3, 'checkpoint': 8,
            'policies': design['policies'], 'maximum_measured_seconds': maximum_seconds,
            'identities': [{k: r[k] for k in ('task_id', 'attempt_id', 'seed')} for task in selected for r in grouped[task]],
            'inputs': {p: file_hash(root / p) for p in ('runs/candidate_bank/pilot_v0.2/trajectories.jsonl',
                      'runs/transfer/protocol_v0.2/models.joblib', 'configs/protocol_v0.2.yaml',
                      'src/slm_steering/verifier.py', 'src/slm_steering/research/generation.py')},
            'interpretation': 'Descriptive known-task end-to-end measurements, N=3; no new primary hypothesis or robust acceleration claim'}
    plan_path = output / 'plan.json'
    if plan_path.exists():
        previous_plan = json.loads(plan_path.read_text(encoding='utf-8'))
        if any(previous_plan[k] != plan[k] for k in ('source_sha256', 'identities', 'inputs', 'modes')):
            raise ValueError('Existing operational plan differs')
    else:
        write_json(plan_path, plan)
    from slm_steering.datasets import CodingProblem
    from slm_steering.research.evaluation import EvalPlusClient
    from slm_steering.verifier import build_candidate_source, normalize_completion
    engine = PrefixGenerator(config)
    if engine.template_hash != bundle['prompt_template_hash']:
        raise ValueError('Operational prompt template differs from the frozen scorer bundle')
    hardware = engine.hardware()
    rows_path = output / 'task_runs.jsonl'
    rows = [json.loads(line) for line in rows_path.read_text(encoding='utf-8').splitlines()] if rows_path.exists() else []
    done = {(r['task_id'], r['repetition'], r['mode']) for r in rows}
    preflight = []
    with EvalPlusClient('humaneval', wsl_python='/home/joaof/.venvs/slm-evalplus-v031/bin/python') as verifier:
        task_metadata = {task['task_id']: task for task in verifier.dataset['tasks']}
        for task in selected:
            if not verifier.request({'operation': 'verify', 'task_id': task, 'canonical_control': True})['passed']:
                raise ValueError('Canonical checker control failed')
        # Warm all paths and check archived token streams and the selected hidden state.
        reference = grouped[selected[0]][0]
        bank = root / 'runs/candidate_bank/pilot_v0.2'
        for mode in MODES:
            sample = generate(engine, reference, mode, bundle, design['policies'])
            if sample['token_ids'] != reference['token_ids'][:len(sample['token_ids'])]:
                raise ValueError('Operational warmup differs from archived tokens')
            if '8' in sample['captured']:
                with np.load(bank / reference['activation_path'], allow_pickle=False) as arrays:
                    index = bundle['layers'].index(bundle['models'][('HB', 8)]['layer'])
                    if not np.array_equal(sample['captured']['8'], arrays['8'][index]):
                        raise ValueError('Selective checkpoint state differs from archived state')
            preflight.append({'mode': mode, 'tokens_match': True, 'aborted': sample['aborted']})
        write_json(output / 'preflight.json', {'created_at': now(), 'hardware': hardware,
                   'checks': preflight, 'warmup_excluded': True, 'canonical_controls': len(selected)})
        measured_start = time.monotonic()
        timed_out = False
        for task_index, task in enumerate(selected):
            for repetition in range(3):
                if time.monotonic() - measured_start >= maximum_seconds:
                    timed_out = True
                    break
                rotation = (task_index + repetition) % len(MODES)
                order = MODES[rotation:] + MODES[:rotation]
                for mode in order:
                    if (task, repetition, mode) in done:
                        continue
                    engine._sync()
                    start = time.perf_counter()
                    attempts, tokens, false_aborts = 0, 0, 0
                    generation = prefill = capture = classification = verification = parsing = 0.
                    passed = False
                    decisions = []
                    for record in grouped[task]:
                        sample = generate(engine, record, mode, bundle, design['policies'])
                        if sample['token_ids'] != record['token_ids'][:len(sample['token_ids'])]:
                            raise ValueError('Measured stream differs from archived tokens')
                        if not sample['aborted'] and sample['token_ids'] != record['token_ids']:
                            raise ValueError('Natural completion is incomplete')
                        attempts += 1
                        tokens += len(sample['token_ids'])
                        generation += sample['generation_seconds']
                        prefill += sample['prefill_seconds']
                        capture += sample['capture_seconds']
                        classification += sample['classification_seconds']
                        decisions.append({'attempt_id': record['attempt_id'], 'aborted': sample['aborted'],
                                          'probability': sample['probability'], 'tokens': len(sample['token_ids'])})
                        if sample['aborted']:
                            false_aborts += int(record['passed'])  # diagnostic after the causal decision
                            continue
                        parse_start = time.perf_counter()
                        problem = CodingProblem(task_id=task, prompt=record['prompt'], entry_point=task_metadata[task]['entry_point'], test='')
                        code = build_candidate_source(problem, sample['raw_completion'])
                        empty = not normalize_completion(sample['raw_completion']).strip()
                        parsing += time.perf_counter() - parse_start
                        verify_start = time.perf_counter()
                        verdict = verifier.verify(task, code)
                        verification += time.perf_counter() - verify_start
                        passed = bool(verdict['passed'] and not empty)
                        if passed != record['passed']:
                            raise ValueError('Functional result differs from archived pilot')
                        if passed:
                            break
                    engine._sync()
                    row = {'task_id': task, 'repetition': repetition, 'mode': mode, 'mode_order': list(order),
                           'total_seconds': time.perf_counter() - start, 'generation_seconds': generation,
                           'prefill_seconds': prefill, 'capture_seconds': capture,
                           'classification_seconds': classification, 'verification_rpc_seconds': verification,
                           'parsing_seconds': parsing, 'tokens': tokens, 'attempts': attempts, 'resolved': passed,
                           'false_aborts': false_aborts, 'decisions': decisions}
                    row['generation_remainder_seconds'] = generation - prefill - capture - classification
                    row['other_task_seconds'] = row['total_seconds'] - generation - verification - parsing
                    if min(row['generation_remainder_seconds'], row['other_task_seconds']) < -1e-6:
                        raise ValueError('Nonadditive host timing spans')
                    with rows_path.open('a', encoding='utf-8') as stream:
                        stream.write(canonical(row) + '\n')
                        stream.flush()
                    rows.append(row)
                    print(f"operational {task} repeat={repetition} mode={mode} seconds={row['total_seconds']:.3f}", flush=True)
            if timed_out:
                break
    frame = pd.DataFrame([{k: v for k, v in row.items() if k not in ('decisions', 'mode_order')} for row in rows])
    frame.to_csv(output / 'task_runs.csv', index=False)
    summaries = []
    for mode, part in frame.groupby('mode', sort=False):
        summaries.append({'mode': mode, 'runs': len(part), 'tasks': part.task_id.nunique(),
            **{column: {'mean': float(part[column].mean()), 'median': float(part[column].median()),
                        'min': float(part[column].min()), 'max': float(part[column].max())}
               for column in ('total_seconds', 'generation_seconds', 'prefill_seconds', 'capture_seconds',
                              'classification_seconds', 'generation_remainder_seconds', 'other_task_seconds',
                              'verification_rpc_seconds', 'parsing_seconds', 'tokens')},
            'resolved_runs': int(part.resolved.sum()), 'false_aborts': int(part.false_aborts.sum())})
    within_task_variability = []
    for (task, mode), part in frame.groupby(['task_id', 'mode']):
        within_task_variability.append({'task_id': task, 'mode': mode, 'repetitions': len(part),
            'total_mean_seconds': float(part.total_seconds.mean()),
            'total_sample_sd_seconds': float(part.total_seconds.std()) if len(part) > 1 else None,
            'total_min_seconds': float(part.total_seconds.min()), 'total_max_seconds': float(part.total_seconds.max())})
    pd.DataFrame(within_task_variability).to_csv(output / 'within_task_timing_variability.csv', index=False)
    result = {'created_at': now(), 'status': 'completed_descriptive' if len(rows) == 72 else 'bounded_partial_descriptive',
              'source_sha256': source_hash, 'runs': len(rows), 'tasks': len(selected), 'maximum_attempts': 3,
              'hardware': engine.hardware(), 'summaries': summaries, 'all_streams_and_labels_match_archive': True,
              'within_task_repetition_variability': within_task_variability,
              'timing_scope': 'Total includes prefill, decoding, gate, parsing and warm-checker RPC; excludes model loading and warmup',
              'components': 'Prefill/capture/classification/remainder partition the generation wall span; generation+parsing+verification+other partition total. Copy can wait for preceding GPU work; spans are not isolated kernel costs.',
              'limitations': 'Six known pilot tasks, three repeated timings per mode, fixed N=3. Descriptive variability, no robust speedup inference, no RTX4060 extrapolation.',
              'inputs': {'reports/excellence/operational_v1/plan.json': file_hash(plan_path)},
              'artifacts': {p.relative_to(root).as_posix(): file_hash(p) for p in (rows_path, output / 'task_runs.csv', output / 'preflight.json',
                                                                                output / 'within_task_timing_variability.csv')}}
    write_json(output / 'results.json', result)
    lines = ['# Latência operacional descritiva', '', result['limitations'], '', result['timing_scope'], '', result['components'], '',
             '| Modo | Execuções | Tempo total médio (s) | Mín–máx (s) | Geração (s) | Verificador RPC (s) | Tokens médios |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        total = row['total_seconds']
        lines.append(f"| {row['mode']} | {row['runs']} | {total['mean']:.3f} | {total['min']:.3f}–{total['max']:.3f} | {row['generation_seconds']['mean']:.3f} | {row['verification_rpc_seconds']['mean']:.3f} | {row['tokens']['mean']:.1f} |")
    lines += ['', 'GPU: ' + hardware['device'] + '. Dados por tarefa, repetição e decisão estão nos JSONL/CSV; o plano precedeu as medições.',
              'Resolver menos tarefas pode reduzir latência. Compare cobertura e tentativas junto com tempo; estas médias não demonstram ganho operacional robusto.', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    return result
