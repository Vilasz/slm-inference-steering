"""Model-specific tokenization and selective, causal t=32 instrumentation."""
from __future__ import annotations

import random
import re
import time

import numpy as np

from slm_steering.final_extension.common import canonical, digest
from slm_steering.final_extension.preparation import MODELS
from slm_steering.generation import render_code_prompt
from slm_steering.research.generation import PrefixGenerator, sampling_statistics
from slm_steering.activations.hooks import resolve_transformer_layers, ActivationHookSet


def configuration(name):
    spec = MODELS[name]
    return {'model': {**spec, 'tokenizer_revision': spec['revision'], 'device': 'cuda', 'dtype': 'bfloat16',
                      'linear_compute_dtype': 'float32', 'attention_backend': 'eager'},
            'resources': {'cpu_threads': 4}, 'decoding': {'top_p': .95, 'max_new_tokens': 512},
            'activations': {'depth_fractions': [.75], 'checkpoints': [32]}}


class ExtensionGenerator(PrefixGenerator):
    def __init__(self, name):
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Frozen BF16 CUDA procedure unavailable; document a prospective amendment')
        super().__init__(configuration(name))
        self.name = name
        self.eos_ids = [self.tokenizer.eos_token_id] if name == 'deepseek' else self.eos_ids
        self.template_hash = digest(canonical({'template': self.tokenizer.chat_template, 'eos': self.eos_ids,
                                               'special_tokens': 'template supplies BOS; encode(add_special_tokens=False)',
                                               'rendered': render_code_prompt(self.tokenizer, '<TASK>')}))

    def encode_prompt(self, prompt):
        rendered = render_code_prompt(self.tokenizer, prompt)
        # DeepSeek's official chat template already emits BOS. encode() defaults
        # to adding BOS again for its LlamaTokenizerFast; never duplicate it.
        ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        if self.name == 'deepseek':
            if ids[0] != self.tokenizer.bos_token_id or ids[:2] == [ids[0], ids[0]]:
                raise ValueError('DeepSeek BOS/template mismatch')
            if not rendered.rstrip().endswith('### Response:'):
                raise ValueError('DeepSeek generation prompt mismatch')
        if self.name == 'qwen' and ids != self.tokenizer.encode(rendered):
            raise ValueError('Qwen origin reuse requires identical historical tokenization')
        return rendered, ids

    def generate(self, prompt, seed, temperature=1., *, capture_states=True):
        torch = self.torch
        from transformers.cache_utils import DynamicCache
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rendered, prompt_ids = self.encode_prompt(prompt)
        capture, captures, states, handles = True, {}, {}, []
        copy_seconds = 0.
        for layer in self.layers if capture_states else []:
            def hook(_module, _inputs, output, layer=layer):
                nonlocal copy_seconds
                if capture:
                    start = time.perf_counter()
                    value = output[0] if isinstance(output, tuple) else output
                    captures[layer] = value[0, -1].detach().float().cpu().numpy().astype(np.float16)
                    copy_seconds += time.perf_counter() - start
            handles.append(resolve_transformer_layers(self.model)[layer].register_forward_hook(hook))
        tokens, lp_values, entropies, margins = [], [], [], []
        self._sync()
        start = time.perf_counter()
        finish = 'length'
        try:
            with torch.inference_mode():
                output = self.model(input_ids=torch.tensor([prompt_ids], device=self.device),
                                    past_key_values=DynamicCache(), use_cache=True)
                if capture_states:
                    states['0'] = np.stack([captures[layer] for layer in self.layers])
                prefill_seconds = time.perf_counter() - start
                for offset in range(1, self.config['decoding']['max_new_tokens'] + 1):
                    probabilities, lp, entropy, margin = sampling_statistics(output.logits[:, -1], temperature, .95)
                    token = torch.multinomial(probabilities, num_samples=1)
                    token_id = int(token.item())
                    tokens.append(token_id)
                    # B observes only the first 32 sampled tokens; never future tokens.
                    if offset <= 32:
                        lp_values.append(float(lp[0, token_id]))
                        entropies.append(float(entropy.item()))
                        margins.append(float(margin.item()))
                    terminal = token_id in self.eos_ids or offset == self.config['decoding']['max_new_tokens']
                    capture = capture_states and (offset == 32 or terminal)
                    output = self.model(input_ids=token, past_key_values=output.past_key_values, use_cache=True)
                    if capture:
                        states['32' if offset == 32 else 'final'] = np.stack([captures[layer] for layer in self.layers])
                    if token_id in self.eos_ids:
                        finish = 'eos'
                        break
                self._sync()
        finally:
            for handle in handles:
                handle.remove()
        if capture_states and 'final' not in states:
            states['final'] = np.stack([captures[layer] for layer in self.layers])
        signature = re.search(r'def\s+\w+\s*\([^)]*\)\s*(?:->[^:]+)?\s*:', prompt, re.DOTALL)
        return {'rendered_prompt': rendered, 'prompt_token_ids': prompt_ids, 'token_ids': tokens,
                'raw_completion': self.tokenizer.decode(tokens, skip_special_tokens=True),
                'generated_tokens': len(tokens), 'finish_reason': finish, 'truncated': finish == 'length',
                'token_logprobs': lp_values, 'token_entropies': entropies, 'token_margins': margins,
                'signature_token_count': len(self.tokenizer.encode(signature.group())) if signature else None,
                'activations': states, 'generation_time_seconds': time.perf_counter() - start,
                'prefill_seconds': prefill_seconds, 'capture_copy_seconds': copy_seconds,
                'eos_token_ids': self.eos_ids}

    def numerical_checks(self, sample):
        torch = self.torch
        ids = sample['prompt_token_ids'] + sample['token_ids']
        prompt_length = len(sample['prompt_token_ids'])
        checks = []
        for t in (0, 32):
            if t > len(sample['token_ids']):
                continue
            position = prompt_length + t - 1
            values = []
            for sequence in (ids[:position + 1], ids):
                with torch.inference_mode(), ActivationHookSet(self.model, self.layers, position) as hooks:
                    self.model.model(input_ids=torch.tensor([sequence], device=self.device), use_cache=False)
                values.append(np.stack([hooks.activations[layer] for layer in self.layers]))
            cached = sample['activations'][str(t)].astype(float)
            for label, other in (('full_sequence', values[1]), ('cached', cached)):
                relative = np.linalg.norm(values[0] - other) / max(np.linalg.norm(values[0]), 1e-12)
                maximum = float(np.max(np.abs(values[0] - other)))
                passed = relative <= .01 and maximum <= .02 + .01 * np.max(np.abs(values[0]))
                checks.append({'checkpoint': t, 'comparison': label, 'relative_l2': float(relative),
                               'maximum_absolute': maximum, 'passed': bool(passed)})
        # Teacher-forcing independently checks distribution and next-token indexing.
        with torch.inference_mode():
            end = prompt_length + min(32, len(sample['token_ids']))
            output = self.model(input_ids=torch.tensor([ids[:end]], device=self.device), use_cache=False)
        for offset, token_id in enumerate(sample['token_ids'][:32]):
            _, lp, entropy, margin = sampling_statistics(output.logits[:, prompt_length + offset - 1], 1., .95)
            for key, value in (('token_logprobs', float(lp[0, token_id])), ('token_entropies', float(entropy)),
                               ('token_margins', float(margin))):
                expected = sample[key][offset]
                checks.append({'token_offset': offset, 'quantity': key,
                               'absolute_difference': abs(value - expected),
                               'passed': bool(np.isfinite(value) and np.isclose(value, expected, atol=.03, rtol=.01))})
        if not all(c['passed'] for c in checks):
            raise ValueError(f'Numerical causality/distribution check failed: {[c for c in checks if not c["passed"]]}')
        return checks
