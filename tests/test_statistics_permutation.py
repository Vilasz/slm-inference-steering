from __future__ import annotations

import unittest

import numpy as np

from slm_steering.statistics.permutation import (
    permutation_test_auc,
    roc_auc_score,
    shuffle_labels_within_groups,
)


class PermutationStatisticsTests(unittest.TestCase):
    def test_auc_score_detects_ordering(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])

        self.assertEqual(roc_auc_score(labels, scores), 1.0)

    def test_shuffle_labels_within_groups_preserves_group_counts(self) -> None:
        labels = np.asarray([0, 1, 0, 1])
        groups = np.asarray(["a", "a", "b", "b"])
        shuffled = shuffle_labels_within_groups(labels, groups=groups, seed=1)

        for group in ["a", "b"]:
            self.assertEqual(int(shuffled[groups == group].sum()), 1)

    def test_permutation_test_returns_p_value(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        result = permutation_test_auc(scores, labels, n_permutations=50, seed=5)

        self.assertGreater(result.observed_score, result.null_mean)
        self.assertGreaterEqual(result.p_value, 0.0)
        self.assertLessEqual(result.p_value, 1.0)


if __name__ == "__main__":
    unittest.main()
