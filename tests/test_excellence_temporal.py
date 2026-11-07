from __future__ import annotations

import copy
import numpy as np

from slm_steering.research.features import observable_features


def test_prefix_observables_do_not_read_suffix_statistics_or_eventual_length():
    record = {'prompt': 'def f(x):\n    return x', 'prompt_token_ids': [1, 2, 3],
              'signature_token_count': 4, 'generated_tokens': 64,
              'token_logprobs': [-.5] * 64, 'token_entropies': [1.] * 64, 'token_margins': [2.] * 64}
    original = observable_features(record, 32)
    altered = copy.deepcopy(record)
    altered['generated_tokens'] = 512
    for name in ('token_logprobs', 'token_entropies', 'token_margins'):
        altered[name] = altered[name][:32] + [1e6] * 480
    np.testing.assert_array_equal(observable_features(altered, 32), original)
