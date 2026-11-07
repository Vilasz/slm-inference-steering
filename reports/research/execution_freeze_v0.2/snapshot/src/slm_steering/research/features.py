from __future__ import annotations

import numpy as np


def observable_features(record: dict, checkpoint: int) -> np.ndarray:
    prompt = record['prompt']
    prompt_features = [len(record['prompt_token_ids']), record.get('signature_token_count'),
                       int('import ' in prompt), int('>>>' in prompt), int('```' in prompt)]
    if checkpoint == 0:
        return np.array([np.nan if value is None else value for value in prompt_features], dtype=float)
    if record['generated_tokens'] < checkpoint:
        raise ValueError('Short completion cannot enter a later checkpoint risk set')
    nll = -np.asarray(record['token_logprobs'][:checkpoint], dtype=float)
    entropy = np.asarray(record['token_entropies'][:checkpoint], dtype=float)
    margins = np.array([np.nan if x is None else x for x in record['token_margins'][:checkpoint]])
    mean_margin = np.nanmean(margins) if np.any(np.isfinite(margins)) else np.nan
    return np.array([np.nan if value is None else value for value in prompt_features] +
                    [nll.mean(), nll[-1], entropy.mean(), entropy.max(), mean_margin])
