"""Paired capture-overhead benchmark on previously examined pilot candidates only."""
from __future__ import annotations

import json
import platform
import random
import re
import time
from pathlib import Path

import numpy as np

from slm_steering.excellence.audit import assert_preserved
from slm_steering.research.generation import PrefixGenerator, sampling_statistics
from slm_steering.research.protocol import digest, file_hash, load_config, now, read_bank, write_json


def selective_generate(engine, prompt, seed, temperature):
    """Same kernels and token order; copy states to CPU only when a stored state is needed."""
    from transformers.cache_utils import DynamicCache
    from slm_steering.activations.hooks import resolve_transformer_layers
    from slm_steering.generation import render_code_prompt
    torch = engine.torch
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    prompt_ids = engine.tokenizer.encode(render_code_prompt(engine.tokenizer, prompt))
    inputs = torch.tensor([prompt_ids], device=engine.device)
    captures, checkpoints, tokens, lp_values, entropies, margins = {}, {}, [], [], [], []
    capture = True
    copy_seconds = 0.0
    handles = []
    for layer in engine.layers:
        def hook(_module, _inputs, output, layer=layer):
            nonlocal copy_seconds
            if capture:
                start = time.perf_counter()
                hidden = output[0] if isinstance(output, tuple) else output
                captures[layer] = hidden[0, -1].detach().float().cpu().numpy().astype(np.float16)
                copy_seconds += time.perf_counter() - start
        handles.append(resolve_transformer_layers(engine.model)[layer].register_forward_hook(hook))
    engine._sync()
    start = time.perf_counter()
    finish = 'length'
    try:
        with torch.inference_mode():
            output = engine.model(input_ids=inputs, past_key_values=DynamicCache(), use_cache=True)
            checkpoints['0'] = np.stack([captures[layer] for layer in engine.layers])
            for offset in range(1, engine.config['decoding']['max_new_tokens'] + 1):
                probabilities, lp, entropy, margin = sampling_statistics(output.logits[:, -1], temperature,
                                                                          engine.config['decoding']['top_p'])
                token = torch.multinomial(probabilities, num_samples=1)
                token_id = int(token.item())
                tokens.append(token_id)
                lp_values.append(float(lp[0, token_id].item()))
                entropies.append(float(entropy.item()))
                margins.append(float(margin.item()))
                terminal = token_id in engine.eos_ids or offset == engine.config['decoding']['max_new_tokens']
                capture = terminal or offset in engine.config['activations']['checkpoints']
                output = engine.model(input_ids=token, past_key_values=output.past_key_values, use_cache=True)
                if offset in engine.config['activations']['checkpoints']:
                    checkpoints[str(offset)] = np.stack([captures[layer] for layer in engine.layers])
                if token_id in engine.eos_ids:
                    finish = 'eos'
                    break
            checkpoints['final'] = np.stack([captures[layer] for layer in engine.layers])
            engine._sync()
    finally:
        for handle in handles:
            handle.remove()
    signature = re.search(r'def\s+\w+\s*\([^)]*\)\s*(?:->[^:]+)?\s*:', prompt, re.DOTALL)
    return {'token_ids': tokens, 'generated_tokens': len(tokens), 'token_logprobs': lp_values,
            'token_entropies': entropies, 'token_margins': margins, 'prompt_token_ids': prompt_ids,
            'raw_completion': engine.tokenizer.decode(tokens, skip_special_tokens=True),
            'finish_reason': finish, 'truncated': finish == 'length', 'activations': checkpoints,
            'signature_token_count': len(engine.tokenizer.encode(signature.group())) if signature else None,
            'generation_time_seconds': time.perf_counter() - start, 'capture_copy_seconds': copy_seconds}


def run(root: Path):
    assert_preserved(root)
    directory = root / 'reports/excellence/hardware'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'results.json').exists():
        archived = json.loads((directory / 'results.json').read_text(encoding='utf-8'))
        if archived['status'] == 'completed' and archived['pairs'] == 50 and archived['all_exact_tokens_and_states']:
            print('Completed archived technical benchmark retained; executed source hash is in hardware/plan.json', flush=True)
            return archived
    config = load_config(root / 'configs/protocol_v0.2.yaml')
    records = read_bank(root / 'runs/candidate_bank/pilot_v0.2/trajectories.jsonl')
    records = [r for r in records if r['decoding_config']['temperature'] == 1.0]
    records.sort(key=lambda r: (r['generated_tokens'], r['task_id'], r['attempt_id']))
    chosen = []
    for indices in np.array_split(np.arange(len(records)), 5):
        candidates = sorted([records[i] for i in indices], key=lambda r: digest(f"excellence-technical-v1:{r['task_id']}:{r['attempt_id']}"))
        chosen.extend(candidates[:10])
    plan = {'created_at': now(), 'status': 'technical_existing_pilot_only', 'pairs': 50,
            'selection': 'Five empirical length strata, ten SHA256-selected candidates per stratum; no correctness filtering',
            'config_sha256': file_hash(root / 'configs/protocol_v0.2.yaml'),
            'identities': [{'task_id': r['task_id'], 'attempt_id': r['attempt_id'], 'seed': r['seed']} for r in chosen],
            'source_sha256': file_hash(Path(__file__)),
            'interpretation': 'Instrumented paired capture implementation benchmark, not a claim of policy latency savings'}
    path = directory / 'plan.json'
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if any(previous[k] != plan[k] for k in ('identities', 'config_sha256', 'source_sha256')):
            raise ValueError('Technical benchmark plan changed; use a new version')
    else:
        write_json(path, plan)
    import torch
    engine = PrefixGenerator(config)
    hardware = {**engine.hardware(), 'python': platform.python_version(), 'torch': torch.__version__,
                'bf16_supported': torch.cuda.is_bf16_supported(), 'requested_gpu': 'RTX 4060',
                'requested_gpu_present': '4060' in engine.hardware()['device']}
    write_json(directory / 'hardware.json', hardware)
    # Warm both paths on a known, short pilot prompt; exclude model loading and warmup.
    for method in (engine.generate, lambda p, s, t: selective_generate(engine, p, s, t)):
        method(chosen[0]['prompt'], chosen[0]['seed'], 1.0)
    rows = []
    rows_path = directory / 'pairs.jsonl'
    if rows_path.exists():
        rows = [json.loads(line) for line in rows_path.read_text(encoding='utf-8').splitlines()]
    for position, record in enumerate(chosen):
        if position < len(rows):
            continue
        results, timings = {}, {}
        order = ['original', 'selective'] if position % 2 == 0 else ['selective', 'original']
        for name in order:
            events = []
            original_forward = engine.model.forward
            def timed_forward(*args, **kwargs):
                before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                before.record()
                value = original_forward(*args, **kwargs)
                after.record()
                events.append((before, after))
                return value
            engine.model.forward = timed_forward
            try:
                sample = engine.generate(record['prompt'], record['seed'], 1.0) if name == 'original' else selective_generate(engine, record['prompt'], record['seed'], 1.0)
            finally:
                engine.model.forward = original_forward
            results[name] = sample
            milliseconds = [a.elapsed_time(b) for a, b in events]
            timings[name] = {'generation_seconds': sample['generation_time_seconds'],
                             'prefill_gpu_seconds': milliseconds[0] / 1000,
                             'decode_forward_gpu_seconds': sum(milliseconds[1:]) / 1000,
                             'decode_forward_gpu_seconds_per_token': sum(milliseconds[1:]) / (1000 * sample['generated_tokens']),
                             'capture_copy_seconds': sample.get('capture_copy_seconds')}
        if results['original']['token_ids'] != results['selective']['token_ids'] or results['original']['token_ids'] != record['token_ids']:
            raise ValueError('Technical optimization changed token identities; benchmark halted')
        for checkpoint, state in results['original']['activations'].items():
            if not np.array_equal(state, results['selective']['activations'][checkpoint]):
                raise ValueError('Selective capture changed a stored activation')
        row = {'position': position, 'task_id': record['task_id'], 'attempt_id': record['attempt_id'],
               'seed': record['seed'], 'tokens': len(record['token_ids']), 'order': order,
               'timings': timings, 'exact_tokens_and_states': True,
               'historical_verification_seconds': record['verification_time_seconds']}
        with rows_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
        rows.append(row)
        write_json(directory / 'status.json', {'status': 'running', 'completed_pairs': len(rows), 'expected_pairs': 50,
                                               'updated_at': now(), 'last_task': record['task_id']})
        print(f"technical benchmark {len(rows)}/50; tokens={row['tokens']}", flush=True)
    hardware.update(engine.hardware())
    write_json(directory / 'hardware.json', hardware)
    result = {'status': 'completed', 'created_at': now(), 'pairs': len(rows), 'trajectories_generated': 2 * len(rows),
              'tasks': len({r['task_id'] for r in rows}), 'hardware': hardware,
              'all_exact_tokens_and_states': all(r['exact_tokens_and_states'] for r in rows),
              'timing_scope': plan['interpretation'],
              'mean_original_seconds': float(np.mean([r['timings']['original']['generation_seconds'] for r in rows])),
              'mean_selective_seconds': float(np.mean([r['timings']['selective']['generation_seconds'] for r in rows])),
              'mean_tokens': float(np.mean([r['tokens'] for r in rows])),
              'mean_historical_verification_seconds': float(np.mean([r['historical_verification_seconds'] for r in rows]))}
    assert_preserved(root)
    write_json(directory / 'results.json', result)
    write_json(directory / 'status.json', {'status': 'completed', 'completed_pairs': len(rows), 'updated_at': now()})
