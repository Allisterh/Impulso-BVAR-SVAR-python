# Writing a Custom Prior

Impulso uses `typing.Protocol` for extensibility. You can write your own prior by implementing the `Prior` protocol.

## The Prior protocol

```python
import numpy as np
from impulso.protocols import Prior


class MyPrior:
    def build_priors(self, n_vars: int, n_lags: int, *, sigma: np.ndarray) -> dict: ...
```

Your `build_priors` method must return a dictionary with keys `"B_mu"` and `"B_sigma"`, both NumPy arrays of shape `(n_vars, n_vars * n_lags)`.

- `B_mu`: Prior mean for VAR coefficient matrix
- `B_sigma`: Prior standard deviation for VAR coefficient matrix

`sigma` is required and keyword-only: a per-endogenous-variable scale, shape `(n_vars,)`, that `VAR.fit` computes once via `impulso._conjugate.ar1_residual_sd(data.endog)` and passes to whichever `Prior` it holds. `MinnesotaPrior` uses it to scale each cross-lag prior standard deviation by `sigma[i] / sigma[j]` — see [The Minnesota Prior](../explanation/minnesota-prior.md#scope-and-caveats). A prior with no use for the data's scale still has to accept the argument; it can simply ignore it, as `FlatPrior` does below.

## Example: Flat prior

```python
import numpy as np


class FlatPrior:
    def build_priors(self, n_vars: int, n_lags: int, *, sigma: np.ndarray) -> dict:
        n_coeffs = n_vars * n_lags
        return {
            "B_mu": np.zeros((n_vars, n_coeffs)),
            "B_sigma": np.ones((n_vars, n_coeffs)) * 10.0,
        }
```

## Using your custom prior

```python
from impulso import VAR

spec = VAR(lags=2, prior=FlatPrior())
fitted = spec.fit(data)
```
