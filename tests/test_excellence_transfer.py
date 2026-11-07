import pandas as pd
import pytest

from slm_steering.excellence.transfer import retention_interval


def test_retention_uses_candidate_ratio_and_task_cluster_uncertainty():
    data = pd.DataFrame({'task_id': ['a', 'b', 'b', 'b', 'c'], 'passed': [1, 1, 1, 1, 0],
                         'probability': [.9, .1, .1, .1, .9]})
    result = retention_interval(data, 'probability', .5, replications=100)
    assert result['estimate'] == pytest.approx(.25)  # not equal-task mean recall .5
    assert result['correct'] == 4 and result['retained'] == 1
    assert result == retention_interval(data, 'probability', .5, replications=100)
