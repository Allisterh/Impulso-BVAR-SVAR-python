"""Shared VAR design-matrix construction.

Lag stacking — assembling the lag-major regressor block a VAR likelihood is
built on — used to be implemented twice: once in `VAR`'s PyMC model builder
and once in the conjugate sampler's design helper. `build_lag_design_matrix`
is the single place that layout is decided, so the two estimators (and,
eventually, a VAR embedded in another PyMC model with a latent endogenous
block) cannot silently drift apart on lag order.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pytensor.tensor as pt


def build_lag_design_matrix(
    endog: "np.ndarray | pt.TensorVariable",
    n_lags: int,
    exog: np.ndarray | None = None,
) -> "tuple[np.ndarray | pt.TensorVariable, np.ndarray | pt.TensorVariable, np.ndarray | None]":
    """Build the lag-major VAR design matrix and trimmed response.

    Stacks `n_lags` lagged blocks of `endog` side by side in **lag-major**
    order — every variable's lag 1, then every variable's lag 2, and so on —
    matching the `coeff` coordinate `VAR._build_pymc_model` labels posterior
    draws with (`"L<lag>.<variable>"`). The leading `n_lags` rows of `endog`
    (and of `exog`, when given) are consumed as initial conditions, so both
    the response and the exogenous block are trimmed to the same estimation
    sample the regressors cover.

    Works on a plain `numpy.ndarray` or on a symbolic 2-D PyTensor tensor:
    the lag blocks are built with the same slicing either way, and only the
    final concatenation branches — `numpy.hstack` for an array,
    `pytensor.tensor.concatenate` for a tensor. This lets `endog` be a
    latent block declared elsewhere in a PyMC model rather than observed
    data. PyTensor is imported lazily, inside the symbolic branch, so
    importing this module never pulls it in.

    Args:
        endog: Endogenous data of shape `(T, n_vars)` — a numpy array of
            observed values, or a symbolic 2-D tensor (e.g. a latent block
            of another PyMC model).
        n_lags: Number of lags to stack. Must be `>= 1`.
        exog: Optional exogenous regressors of shape `(T, n_exog)`, trimmed
            alongside `endog`. `None` (the default) if the model has no
            exogenous block.

    Returns:
        Tuple `(Y, X_lag, X_exog)`:

        * `Y`: trimmed response, `endog[n_lags:]`, shape
          `(T - n_lags, n_vars)`.
        * `X_lag`: lag-major stacked regressors, shape
          `(T - n_lags, n_vars * n_lags)`, same array/tensor kind as
          `endog`.
        * `X_exog`: `exog[n_lags:]` if `exog` was given, else `None`.

    Raises:
        ValueError: If `n_lags` is not positive.
    """
    if n_lags < 1:
        raise ValueError(f"n_lags must be positive, got {n_lags}")

    lag_blocks = [endog[n_lags - lag : -lag] for lag in range(1, n_lags + 1)]
    if isinstance(endog, np.ndarray):
        x_lag = np.hstack(lag_blocks)
    else:
        import pytensor.tensor as pt

        x_lag = pt.concatenate(lag_blocks, axis=1)

    y = endog[n_lags:]
    x_exog = exog[n_lags:] if exog is not None else None
    return y, x_lag, x_exog
