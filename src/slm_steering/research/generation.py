from __future__ import annotations

import random
import re
import subprocess
import time

import numpy as np

from slm_steering.activations.hooks import resolve_transformer_layers
from slm_steering.generation import render_code_prompt
from slm_steering.research.protocol import block_indices, canonical, digest


def sampling_statistics(logits, temperature: float, top_p: float):
    """Distribution actually sampled; mirrors HF TopPLogitsWarper, min_tokens_to_keep=1."""
    import torch
    if not torch.isfinite(logits).all():
        raise ValueError('Non-finite model logits; precision/backend must pass hardware validation before sampling')
    scaled = logits.float() / temperature
    sorted_logits, order = torch.sort(scaled, descending=False, dim=-1)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    removed = cumulative <= (1 - top_p)
    removed[..., -1] = False
    remove_original = removed.scatter(-1, order, removed)
    filtered = scaled.masked_fill(remove_original, -torch.inf)
    log_probs = torch.log_softmax(filtered, dim=-1)
    probabilities = log_probs.exp()
    if not torch.isfinite(probabilities).all() or not torch.all(probabilities.sum(dim=-1) > 0):
        raise ValueError('Invalid sampling distribution; no trajectory or label may be recorded')
    entropy = -(probabilities * log_probs.masked_fill(~torch.isfinite(log_probs), 0)).sum(dim=-1)
    # Confidence margin is defined before nucleus filtering, after temperature.
    # It remains finite even when top-p retains exactly one token.
    top = torch.topk(scaled, 2, dim=-1).values
    margin = top[..., 0] - top[..., 1]
    margin = torch.where(torch.isfinite(margin), margin, torch.full_like(margin, torch.nan))
    return probabilities, log_probs, entropy, margin


class PrefixGenerator:
    def __init__(self, config: dict, *, local_files_only: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.config = torch, config
        self.driver_version = None
        model_config = config['model']
        self.device = model_config['device']
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no silent CPU fallback')
        if self.device == 'cuda':
            # Accumulate GEMM reductions in full precision; reduced accumulation
            # can overflow fp16 or introduce shape-dependent bf16 prefix drift.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_fp16_accumulation = False
            self.driver_version = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True, timeout=10).strip()
        torch.set_num_threads(config['resources']['cpu_threads'])
        self.tokenizer = AutoTokenizer.from_pretrained(model_config['id'], revision=model_config['tokenizer_revision'],
                                                       local_files_only=local_files_only, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config['id'], revision=model_config['revision'], torch_dtype=getattr(torch, model_config['dtype']),
            attn_implementation=model_config['attention_backend'], local_files_only=local_files_only,
            trust_remote_code=False).to(self.device).eval()
        if model_config.get('linear_compute_dtype') == 'float32':
            # Keep 16-bit parameter storage within 6 GB, but evaluate dense
            # projections in float32. The temporary weight cast is freed after
            # each projection; no second full model is kept on the GPU.
            for module in self.model.modules():
                if isinstance(module, torch.nn.Linear):
                    def float32_forward(hidden, module=module):
                        return torch.nn.functional.linear(hidden.float(), module.weight.float(),
                            module.bias.float() if module.bias is not None else None)
                    module.forward = float32_forward
        self.layers = block_indices(self.model.config.num_hidden_layers, config['activations']['depth_fractions'])
        self.eos_ids = self.model.generation_config.eos_token_id
        if isinstance(self.eos_ids, int):
            self.eos_ids = [self.eos_ids]
        self.eos_ids = self.eos_ids or [self.tokenizer.eos_token_id]
        self.template_hash = digest(canonical({'chat_template': self.tokenizer.chat_template,
                                              'renderer': render_code_prompt(self.tokenizer, '<TASK>'),
                                              'eos_ids': self.eos_ids}))
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()

    def _sync(self):
        if self.device == 'cuda':
            self.torch.cuda.synchronize()

    def generate(self, prompt: str, seed: int, temperature: float, abort=None) -> dict:
        torch = self.torch
        from transformers.cache_utils import DynamicCache
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed_all(seed)
        rendered = render_code_prompt(self.tokenizer, prompt)
        prompt_ids = self.tokenizer.encode(rendered)
        signature = re.search(r'def\s+\w+\s*\([^)]*\)\s*(?:->[^:]+)?\s*:', prompt, re.DOTALL)
        inputs = torch.tensor([prompt_ids], device=self.device)
        tokens, logprobs, entropies, margins = [], [], [], []
        captures = {}
        handles = []
        blocks = resolve_transformer_layers(self.model)
        for layer in self.layers:
            def hook(_module, _inputs, output, layer=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                captures[layer] = hidden[0, -1].detach().float().cpu().numpy().astype(np.float16)
            handles.append(blocks[layer].register_forward_hook(hook))
        checkpoints = {}
        start = time.perf_counter()
        self._sync()
        finish = 'length'
        try:
            with torch.inference_mode():
                output = self.model(input_ids=inputs, past_key_values=DynamicCache(), use_cache=True)
                checkpoints['0'] = np.stack([captures[layer] for layer in self.layers])
                for offset in range(1, self.config['decoding']['max_new_tokens'] + 1):
                    probabilities, lp, entropy, margin = sampling_statistics(
                        output.logits[:, -1], temperature, self.config['decoding']['top_p'])
                    token = torch.multinomial(probabilities, num_samples=1)
                    token_id = int(token.item())
                    tokens.append(token_id)
                    logprobs.append(float(lp[0, token_id].item()))
                    entropies.append(float(entropy.item()))
                    value = float(margin.item())
                    margins.append(value if np.isfinite(value) else None)
                    # Feed the sampled token before capturing its hidden state. The preceding
                    # model output predicted this token and is NOT its activation.
                    output = self.model(input_ids=token, past_key_values=output.past_key_values, use_cache=True)
                    if offset in self.config['activations']['checkpoints']:
                        checkpoints[str(offset)] = np.stack([captures[layer] for layer in self.layers])
                    if token_id in self.eos_ids:
                        finish = 'eos'
                        break
                    if offset == self.config['decoding']['max_new_tokens']:
                        break
                    if abort is not None and abort(offset, captures, logprobs, entropies, margins):
                        finish = 'abort'
                        break
                checkpoints['final'] = np.stack([captures[layer] for layer in self.layers])
                self._sync()
        finally:
            for handle in handles:
                handle.remove()
        return {'raw_completion': self.tokenizer.decode(tokens, skip_special_tokens=True),
                'rendered_prompt': rendered, 'prompt_token_ids': prompt_ids, 'token_ids': tokens,
                'signature_token_count': len(self.tokenizer.encode(signature.group())) if signature else None,
                'eos_token_ids': self.eos_ids,
                'token_logprobs': logprobs, 'token_entropies': entropies, 'token_margins': margins,
                'generated_tokens': len(tokens), 'finish_reason': finish, 'truncated': finish == 'length',
                'generation_time_seconds': time.perf_counter() - start, 'activations': checkpoints}

    def invariance_test(self, sample: dict) -> dict:
        """Compare real block states at prefix positions with full-sequence states."""
        torch = self.torch
        from slm_steering.activations.hooks import ActivationHookSet
        from slm_steering.research.protocol import token_position
        all_ids = sample['prompt_token_ids'] + sample['token_ids']
        checks = []
        tolerance = self.config['activations']
        for checkpoint in (0, 8, 16, 32, 64, 'final'):
            if checkpoint != 'final' and int(checkpoint) > len(sample['token_ids']):
                continue
            position = token_position(len(sample['prompt_token_ids']), len(sample['token_ids']), checkpoint)
            states = []
            for ids in (all_ids[:position + 1], all_ids):
                with torch.inference_mode(), ActivationHookSet(self.model, self.layers, position) as hooks:
                    # Avoid allocating [full sequence, vocabulary] LM logits during extraction.
                    self.model.model(input_ids=torch.tensor([ids], device=self.device), use_cache=False)
                states.append(np.stack([hooks.activations[layer] for layer in self.layers]))
            comparisons = []
            for name, other in [('full_forward', states[1]), ('cached_generation', sample['activations'][str(checkpoint)].astype(float))]:
                difference = states[0] - other
                l2 = np.linalg.norm(difference, axis=1) / np.maximum(np.linalg.norm(states[0], axis=1), 1e-12)
                maximum = np.max(np.abs(difference), axis=1)
                limit = tolerance['invariance_atol'] + tolerance['invariance_rtol'] * np.max(np.abs(states[0]), axis=1)
                ok = bool(np.all(l2 <= tolerance['invariance_rtol']) and np.all(maximum <= limit))
                comparisons.append({'comparison': name, 'passed': ok, 'relative_l2_by_layer': l2.tolist(),
                                    'max_absolute_difference_by_layer': maximum.tolist(), 'maximum_limits': limit.tolist()})
            checks.append({'checkpoint': checkpoint, 'passed': all(c['passed'] for c in comparisons), 'comparisons': comparisons})
        result = {'passed': all(check['passed'] for check in checks), 'checks': checks,
                  'atol': tolerance['invariance_atol'], 'rtol': tolerance['invariance_rtol'],
                  'metric': tolerance['invariance_metric']}
        if not result['passed']:
            raise ValueError(f'Causal prefix invariance failed: {result}')
        return result

    def hardware(self) -> dict:
        torch = self.torch
        if self.device != 'cuda':
            return {'device': self.device}
        props = torch.cuda.get_device_properties(0)
        return {'device': props.name, 'memory_total_mib': props.total_memory / 2**20,
                'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20, 'cuda': torch.version.cuda,
                'driver_version': self.driver_version, 'weight_storage_dtype': self.config['model']['dtype'],
                'linear_compute_dtype': self.config['model'].get('linear_compute_dtype', self.config['model']['dtype'])}
