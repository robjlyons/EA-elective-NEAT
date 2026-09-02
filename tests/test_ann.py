import numpy as np
import pytest

from prettyNEAT.ann import weightedRandom


@pytest.mark.parametrize(
    ("weights", "expected_probabilities"),
    [
        ([2, 2, 2], [1 / 3, 1 / 3, 1 / 3]),
        ([0, 0, 0], [1 / 3, 1 / 3, 1 / 3]),
        ([-3, -1, 1], [0, 1 / 3, 2 / 3]),
        ([0, 7, 0], [0, 1, 0]),
    ],
)
def test_weighted_random_uses_normalized_probabilities(
    monkeypatch, weights, expected_probabilities
):
    calls = []

    def choice(size, p):
        calls.append((size, p))
        return 1

    monkeypatch.setattr(np.random, "choice", choice)

    assert weightedRandom(weights) == 1
    assert calls[0][0] == len(weights)
    np.testing.assert_allclose(calls[0][1], expected_probabilities)


@pytest.mark.parametrize("weights", [[], [np.nan], [np.inf], [-np.inf]])
def test_weighted_random_rejects_empty_or_nonfinite_weights(weights):
    with pytest.raises(ValueError):
        weightedRandom(weights)
