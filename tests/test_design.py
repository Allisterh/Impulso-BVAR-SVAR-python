"""Tests for the shared lag design-matrix builder (issue 02).

Gates:

* Numpy stacking matches an independently derived reference (via
  `pandas.DataFrame.shift`, not the module's own slicing), for a range of
  lag orders, with and without exog.
* The symbolic (PyTensor) path evaluates to the same arrays as the numpy
  path.
* `n_lags < 1` is rejected.
* The builder is exported from the public `impulso` namespace.
"""

import numpy as np
import pandas as pd
import pytest

LAG_ORDERS = [1, 2, 4]


def _reference_lag_block(endog: np.ndarray, n_lags: int, lag: int) -> np.ndarray:
    """`endog` shifted by `lag` rows via pandas, trimmed to the estimation sample.

    Deliberately independent of the module's own numpy-slicing
    implementation: `DataFrame.shift` is a different mechanism for producing
    the same lagged values.
    """
    return pd.DataFrame(endog).shift(lag).iloc[n_lags:].to_numpy()


def _reference_design(
    endog: np.ndarray, n_lags: int, exog: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Independent reference `(Y, X_lag, X_exog)` for `build_lag_design_matrix`."""
    y = endog[n_lags:]
    x_lag = np.hstack([_reference_lag_block(endog, n_lags, lag) for lag in range(1, n_lags + 1)])
    x_exog = exog[n_lags:] if exog is not None else None
    return y, x_lag, x_exog


@pytest.mark.parametrize("n_lags", LAG_ORDERS)
@pytest.mark.parametrize("with_exog", [False, True])
def test_numpy_stacking_matches_reference(rng, n_lags, with_exog):
    from impulso._design import build_lag_design_matrix  # ty: ignore[unresolved-import]

    endog = rng.standard_normal((30, 3))
    exog = rng.standard_normal((30, 2)) if with_exog else None

    y, x_lag, x_exog = build_lag_design_matrix(endog, n_lags, exog)
    exp_y, exp_x_lag, exp_x_exog = _reference_design(endog, n_lags, exog)

    np.testing.assert_array_equal(y, exp_y)
    assert x_lag.shape == (30 - n_lags, 3 * n_lags)
    np.testing.assert_array_equal(x_lag, exp_x_lag)
    if with_exog:
        np.testing.assert_array_equal(x_exog, exp_x_exog)
    else:
        assert x_exog is None


@pytest.mark.parametrize("n_lags", LAG_ORDERS)
def test_symbolic_path_matches_numpy(rng, n_lags):
    """A symbolic 2-D input takes the `pytensor.tensor.concatenate` branch."""
    import pytensor.tensor as pt

    from impulso._design import build_lag_design_matrix  # ty: ignore[unresolved-import]

    endog = rng.standard_normal((30, 3))
    y_np, x_lag_np, _ = build_lag_design_matrix(endog, n_lags)

    endog_sym = pt.matrix("endog")
    y_sym, x_lag_sym, x_exog_sym = build_lag_design_matrix(endog_sym, n_lags)
    assert x_exog_sym is None

    np.testing.assert_allclose(y_sym.eval({endog_sym: endog}), y_np)
    np.testing.assert_allclose(x_lag_sym.eval({endog_sym: endog}), x_lag_np)


def test_rejects_non_positive_n_lags(rng):
    from impulso._design import build_lag_design_matrix  # ty: ignore[unresolved-import]

    endog = rng.standard_normal((10, 2))
    with pytest.raises(ValueError, match="n_lags must be positive"):
        build_lag_design_matrix(endog, 0)


def test_exported_from_public_api():
    import impulso

    assert "build_lag_design_matrix" in impulso.__all__
    assert callable(impulso.build_lag_design_matrix)
