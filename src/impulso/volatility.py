"""Volatility processes for the VAR pipeline.

Defines concrete adapters of the VolatilityProcess Protocol declared in
protocols.py. The constant adapter (Constant) holds today's homoscedastic
manual-Cholesky parameterisation; stochastic adapters live elsewhere
(StochasticVolatility in impulso.sv) and arrive in later phases.

See docs/adr/0001-volatility-process-seam-exposes-cholesky-factor.md.
"""

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import Field

from impulso._base import ImpulsoModel

if TYPE_CHECKING:
    import pytensor.tensor as pt
    import xarray as xr


class InnovationScalePrior(ImpulsoModel):
    """Prior family and scale for one endogenous variable's Cholesky-diagonal entry.

    Used by `Constant.innovation_scale_priors` to override the single shared
    `HalfCauchy(sigma_sd_beta)` with a per-variable family and scale. See that
    field, and the `Constant` class docstring, for what the diagonal entry
    does and does not mean under ADR-0014's manual Cholesky.

    Attributes:
        family: Distribution family for the diagonal entry: `"halfnormal"`
            (`HalfNormal(sigma=scale)`), `"exponential"`
            (`Exponential(scale=scale)`, i.e. mean `scale`, rate `1/scale`),
            or `"halfcauchy"` (`HalfCauchy(beta=scale)` — the same family
            `Constant` uses by default).
        scale: Positive scale parameter. For `"halfnormal"` and
            `"halfcauchy"` this is the distribution's usual scale
            (`sigma` / `beta`). For `"exponential"` this is the
            distribution's *mean*, not its rate — the rate passed to PyMC
            is `1 / scale`.
    """

    family: Literal["halfnormal", "exponential", "halfcauchy"]
    scale: float = Field(gt=0)


class Constant(ImpulsoModel):
    """Homoscedastic volatility — single Σ shared across all time points.

    Lifts today's manual-Cholesky parameterisation from
    `spec.py:_build_pymc_model` into the volatility-process seam:
    HalfCauchy(beta=sigma_sd_beta) on the diagonal scales,
    Normal(mu=0, sigma=tril_offdiag_sigma) on the lower-triangular
    off-diagonals (scaled by the row's diagonal). For `n_vars == 1`
    the off-diagonal block is empty.

    The factor is assembled from primitives rather than with PyMC's
    purpose-built `LKJCholeskyCov` / `LKJCorr` because those are broken on the
    dependency set Impulso supports — an einsum unpacking bug — so the obvious
    built-in is not an option here. See
    docs/adr/0014-manual-cholesky-parameterisation.md.

    The PyMC variable names produced inside `build_pymc_latent`
    (`sigma_sd`, `tril_offdiag`) match today's posterior contents
    exactly so existing identification and downstream code keep working
    unchanged. The `Sigma = L @ L.T` deterministic is registered by
    the caller in `spec.py`, not by the adapter.

    Under ADR-0014's manual Cholesky, row `i`'s Cholesky diagonal entry is
    only exactly that row's innovation standard deviation for the *first*
    variable (`i == 0`): row 0 has no lower-triangular entries, so its
    innovation sd is `sd[0]` exactly. Every other row also carries
    off-diagonal `tril_offdiag` entries, so row `i`'s actual innovation sd is
    `sd[i] * sqrt(1 + sum_j tril_ij**2)`, not `sd[i]` alone. A per-variable
    prior placed on the diagonal (via `innovation_scale_priors`) therefore
    means exactly what it says only for the first variable; for later
    variables it is a prior on one factor of a larger quantity.

    Attributes:
        name: Discriminator key for the registry (always `"constant"`).
        is_time_varying: Always `False` — Σ is shared across t.
        sigma_sd_beta: HalfCauchy scale on diagonal SDs. Ignored when
            `innovation_scale_priors` is set.
        tril_offdiag_sigma: Normal SD on off-diagonal correlation factors.
        innovation_scale_priors: Optional per-variable override of the
            diagonal prior, one `InnovationScalePrior` per endogenous
            variable in `data.endog_names` order. `None` (the default)
            reproduces today's behaviour exactly: a single vectorised
            `HalfCauchy(sigma_sd_beta)` registered as `sigma_sd`, with the
            same log-probability as before this field existed. When set,
            `build_pymc_latent` instead registers one scalar RV per
            variable, named `sigma_sd_0`, `sigma_sd_1`, ... in variable
            order (mixed families are allowed), which are stacked into the
            diagonal — a different posterior variable layout from the
            default, documented here rather than silently changed. The
            length must equal `n_vars`; a mismatch raises `ValueError` from
            `build_pymc_latent` (this model does not know `n_vars` at
            construction time, so the check cannot happen earlier).
    """

    name: Literal["constant"] = "constant"
    is_time_varying: bool = False

    sigma_sd_beta: float = Field(2.5, gt=0)
    tril_offdiag_sigma: float = Field(0.5, gt=0)
    innovation_scale_priors: tuple[InnovationScalePrior, ...] | None = None

    def build_pymc_latent(
        self,
        n_vars: int,
        T: int,
        data: np.ndarray | None = None,
    ) -> "pt.TensorVariable":
        """Register the constant-volatility latent vars in the active PyMC model.

        Lifts the manual-Cholesky parameterisation from the previous
        location in `spec.py:_build_pymc_model`. When `innovation_scale_priors`
        is unset, PyMC variable names (`sigma_sd`, `tril_offdiag`) match the
        prior contents byte-for-byte so existing posterior-consuming code
        keeps working unchanged. When it is set, the diagonal is instead
        registered as one scalar RV per variable (`sigma_sd_0`,
        `sigma_sd_1`, ...) — see `innovation_scale_priors`.

        Args:
            n_vars: Number of endogenous variables.
            T: Number of observations after lag trimming. Ignored for
                constant volatility — kept in the signature for parity
                with stochastic adapters.
            data: Accepted for Protocol parity with stochastic adapters and
                ignored — Σ is data-independent in the constant case.

        Returns:
            Lower-triangular Cholesky factor L of shape (n_vars, n_vars).

        Raises:
            ValueError: `innovation_scale_priors` is set and its length does
                not equal `n_vars`.
        """
        import pymc as pm
        import pytensor.tensor as pt

        if self.innovation_scale_priors is None:
            sd = pm.HalfCauchy("sigma_sd", beta=self.sigma_sd_beta, shape=n_vars)
        else:
            if len(self.innovation_scale_priors) != n_vars:
                raise ValueError(
                    f"Constant.innovation_scale_priors has {len(self.innovation_scale_priors)} "
                    f"entries but the model has {n_vars} endogenous variables. Supply exactly "
                    "one InnovationScalePrior per variable, or omit the field to use the "
                    f"default HalfCauchy(beta={self.sigma_sd_beta}) prior for every variable."
                )
            sd_components = []
            for i, innovation_prior in enumerate(self.innovation_scale_priors):
                var_name = f"sigma_sd_{i}"
                if innovation_prior.family == "halfnormal":
                    sd_components.append(pm.HalfNormal(var_name, sigma=innovation_prior.scale))
                elif innovation_prior.family == "exponential":
                    sd_components.append(pm.Exponential(var_name, scale=innovation_prior.scale))
                else:
                    sd_components.append(pm.HalfCauchy(var_name, beta=innovation_prior.scale))
            sd = pt.stack(sd_components)
        n_tril = n_vars * (n_vars - 1) // 2
        L = pt.zeros((n_vars, n_vars))
        L = pt.set_subtensor(L[np.diag_indices(n_vars)], sd)
        if n_tril > 0:
            tril_vals = pm.Normal("tril_offdiag", mu=0, sigma=self.tril_offdiag_sigma, shape=n_tril)
            idx = 0
            for i in range(1, n_vars):
                for j in range(i):
                    L = pt.set_subtensor(L[i, j], tril_vals[idx] * sd[i])
                    idx += 1
        # Expose L as a deterministic so cholesky_at can read it directly
        # from the posterior instead of re-decomposing Σ on every call.
        return pm.Deterministic("L", L)

    def cholesky_at(self, posterior: "xr.Dataset", t: int | None) -> np.ndarray:
        """Return the lower-triangular Cholesky factor of Σ for every draw.

        Reads `posterior["L"]` directly — the factor is registered as a
        deterministic in `build_pymc_latent` so this method does not
        re-decompose Σ. For constant volatility, `t` is ignored.

        Args:
            posterior: An xarray Dataset (typically `idata.posterior`)
                containing `L` of shape (chains, draws, n_vars, n_vars).
            t: Time index. Ignored.

        Returns:
            Cholesky factors of shape (chains, draws, n_vars, n_vars).
        """
        return posterior["L"].values

    def forecast_cholesky_path(
        self,
        posterior: "xr.Dataset",
        steps: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Broadcast the constant Cholesky factor across forecast steps.

        For constant volatility there is nothing to simulate — the forecast
        covariance equals the in-sample covariance. `rng` is accepted for
        signature parity with stochastic adapters and is ignored.

        Args:
            posterior: An xarray Dataset containing `L` of shape
                (chains, draws, n_vars, n_vars). Read via
                `Constant.cholesky_at`, which is the canonical accessor.
            steps: Forecast horizon.
            rng: Unused.

        Returns:
            Cholesky factor path of shape (chains, draws, steps, n_vars, n_vars).
        """
        L = self.cholesky_at(posterior, t=None)  # (C, D, n, n)
        return np.broadcast_to(L[:, :, np.newaxis, :, :], (*L.shape[:2], steps, *L.shape[-2:])).copy()

    def cholesky_path(self, posterior: "xr.Dataset", T: int) -> np.ndarray:
        """Broadcast the constant Cholesky factor across all in-sample t.

        For constant volatility there is no per-t variation; this is a
        broadcast convenience for the IdentifiedVAR query layer.

        Args:
            posterior: An xarray Dataset containing `L` of shape
                (chains, draws, n_vars, n_vars). Read via
                `Constant.cholesky_at`, which is the canonical accessor.
            T: In-sample length (after lag trimming).

        Returns:
            Cholesky factor path of shape (chains, draws, T, n_vars, n_vars).
        """
        L = self.cholesky_at(posterior, t=None)  # (C, D, n, n)
        return np.broadcast_to(L[:, :, np.newaxis, :, :], (*L.shape[:2], T, *L.shape[-2:])).copy()
