import numpy as np

from synfire.core.backend import backend


def test_sigmoid_preserves_fractional_values_for_integer_input():
    values = backend.sigmoid(np.array([0, 2], dtype=np.int64))

    np.testing.assert_allclose(values, [0.5, 1.0 / (1.0 + np.exp(-2.0))])
