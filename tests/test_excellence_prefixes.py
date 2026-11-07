import pandas as pd
import pytest

from slm_steering.excellence.prefixes import summarize_prefixes


def test_shared_prefix_pairs_force_ties_and_finished_candidates_are_excluded():
    records = [{'task_id': t, 'attempt_id': j, 'token_ids': tokens, 'generated_tokens': len(tokens), 'passed': y}
               for t, j, tokens, y in [('a', 0, [1, 2, 3], 0), ('a', 1, [1, 2, 4], 1),
                                       ('b', 0, [9, 8, 3], 0), ('b', 1, [9, 7, 4], 1),
                                       ('c', 0, [1, 2], 1)]]
    prediction = pd.DataFrame({'task_id': ['a', 'a', 'b', 'b'], 'attempt_id': [0, 1, 0, 1],
                               'checkpoint': ['2'] * 4, 'HB': [.5, .5, .2, .8]})
    summary, tasks = summarize_prefixes(records, prediction, 2)
    assert summary['tasks'] == summary['mixed_tasks'] == 2
    assert summary['mixed_single_prefix_tasks'] == 1
    assert summary['mean_fraction_unavoidably_tied_pairs'] == .5
    assert all(t['task_id'] != 'c' for t in tasks)
    prediction.loc[1, 'HB'] = .6
    with pytest.raises(ValueError, match='Identical prompt/prefix'):
        summarize_prefixes(records, prediction, 2)
