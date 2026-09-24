"""Prior specifications for VAR models."""

from typing import Literal

import numpy as np
from pydantic import Field

from impulso._base import ImpulsoModel
from impulso._conjugate import ar1_residual_sd, minnesota_dummies


class MinnesotaPrior(ImpulsoModel):
    """Minnesota prior for VAR coefficient shrinkage.

    `tightness` is always held fixed at the value supplied; there is no estimation path
    for it. That is the deliberate contrast with :class:`NIWPrior`, whose ``select`` flag
    estimates the tightness by marginal likelihood — the independent-Normal coefficient
    prior used here has no closed-form marginal likelihood to maximise, so the shrinkage
    stays a modelling choice rather than an estimand.

    Attributes:
        tightness: Overall shrinkage toward prior mean. Must be > 0. Always fixed, never
            estimated from the data.
        decay: How coefficients shrink on longer lags.
        cross_shrinkage: Shrinkage on other variables' lags vs own. 0 = only own lags, 1 = equal.
    """

    tightness: float = Field(0.1, gt=0)
    decay: Literal["harmonic", "geometric"] = "harmonic"
    cross_shrinkage: float = Field(0.5, ge=0, le=1)

    def build_priors(self, n_vars: int, n_lags: int, *, sigma: np.ndarray) -> dict[str, np.ndarray]:
        """Build prior mean and standard deviation arrays for VAR coefficients.

        The prior standard deviation on the coefficient linking lag `l` of
        variable `j` to the equation for variable `i` is scaled by
        `sigma[i] / sigma[j]` — the textbook Minnesota/Litterman cross-lag
        form, and the same ratio the conjugate `NIWPrior` already applies via
        `minnesota_dummies`. On own lags (`i == j`) the ratio is `sigma[i] /
        sigma[i]`, which is 1 for any finite nonzero `sigma[i]`, so this
        leaves the own-lag standard deviations unchanged; it only rescales
        cross-lag entries. That "1" would break down if `sigma[i]` were
        *exactly* zero (`0.0 / 0.0` is `nan`, not 1) — see the warning below
        for why that case cannot reach here. See docs/adr/0015 for why the
        scaling itself is always on, with no opt-out.

        Warning:
            A near-zero but nonzero `sigma[c]` (e.g. `data.endog[:, c]` is
            numerically flat but not exactly constant) is accepted, not
            guarded against: every cross-lag entry in row `c` (`sigma[c]` in
            the numerator) collapses toward zero, and every cross-lag entry
            in every *other* row that references variable `c`'s lag
            (`sigma[c]` in the denominator) blows up instead — the ratio can
            reach `1e12` or more for an otherwise-ordinary numerically
            near-constant column. That is by design (issue 07b): the column
            still varies, so it is still a real, identified variable: it
            just gets an effectively flat, uninformative cross-lag prior
            everywhere but its own equation.

            An exactly-zero or non-finite `sigma[c]` is different in kind,
            not degree — the own-lag entry `B_sigma[c, c]` would be `0.0 /
            0.0`, i.e. `nan`, not the unscaled value the "ratio is 1 on own
            lags" description above promises — and *is* rejected: see Raises
            below. `impulso.data.VARData` also rejects endogenous columns
            that are exactly constant over the whole sample before `sigma`
            is ever computed from them, and `VAR._build_pymc_model` checks
            the `sigma` it computes before calling this method, so both
            guards run ahead of `build_priors` on the normal `VAR.fit` /
            `VAR.prior_predictive` path (issue 07b). The check here exists
            for callers who construct `MinnesotaPrior` and call
            `build_priors` directly, supplying their own `sigma`.

        Args:
            n_vars: Number of endogenous variables.
            n_lags: Number of lags.
            sigma: Per-variable scale, shape `(n_vars,)` — typically
                `impulso._conjugate.ar1_residual_sd(data.endog)`. Required and
                keyword-only (see `Prior.build_priors`). Every entry must be
                finite and strictly positive.

        Returns:
            Dictionary with keys 'B_mu' and 'B_sigma' as numpy arrays.

        Raises:
            ValueError: If `sigma` does not have length `n_vars`, or if any
                entry of `sigma` is zero, negative, or non-finite (issue 07b).
        """
        sigma = np.asarray(sigma, dtype=float)
        if sigma.shape != (n_vars,):
            raise ValueError(f"sigma must have shape ({n_vars},) to match n_vars={n_vars}, got shape {sigma.shape}")

        bad = np.flatnonzero(~np.isfinite(sigma) | (sigma <= 0.0))
        if bad.size:
            values = ", ".join(f"sigma[{i}]={sigma[i]!r}" for i in bad)
            raise ValueError(
                f"sigma must be finite and strictly positive for every variable, got {values}. A zero or "
                "non-finite entry usually means the corresponding endogenous column is constant (or "
                "otherwise degenerate): the cross-lag ratio sigma[i]/sigma[j] this method scales by "
                "(docs/adr/0015) would collapse that column's own row toward zero, send every other row's "
                "coefficient on its lag to inf, and turn the own-lag entry into 0.0 / 0.0 = nan. "
                "impulso.data.VARData rejects exactly-constant endogenous columns for this reason; fix "
                "sigma (or the data it was derived from) upstream."
            )

        n_coeffs = n_vars * n_lags

        # B_mu: identity on the first lag block, zero elsewhere
        B_mu = np.zeros((n_vars, n_coeffs))
        B_mu[np.arange(n_vars), np.arange(n_vars)] = 1.0

        # Lag decay per column: each lag's decay repeated n_vars times
        lags = np.arange(1, n_lags + 1)
        lag_decay = 1.0 / lags if self.decay == "harmonic" else 1.0 / lags**2
        decay_per_col = np.repeat(lag_decay, n_vars)  # (n_coeffs,)

        # Own vs cross mask: 1.0 on own-variable columns, cross_shrinkage elsewhere
        col_var = np.arange(n_coeffs) % n_vars
        is_own = col_var[np.newaxis, :] == np.arange(n_vars)[:, np.newaxis]
        cross_mask = np.where(is_own, 1.0, self.cross_shrinkage)

        # sigma[i] / sigma[j]: 1.0 on own lags (i == j), the Litterman ratio on cross
        # lags. sigma == 0 (a constant endogenous column) would collapse row c toward
        # zero and blow up column c in every other row -- but that is rejected above
        # before this line runs, so sigma is guaranteed finite and strictly positive
        # here. A near-zero (but nonzero) sigma[c] still lands the ratio in the 1e12+
        # range; that is accepted by design, not guarded. See docs/adr/0015, the
        # class docstring, and the Warning above (issue 07b).
        scale_ratio = sigma[:, np.newaxis] / sigma[col_var][np.newaxis, :]

        B_sigma = self.tightness * decay_per_col[np.newaxis, :] * cross_mask * scale_ratio

        return {"B_mu": B_mu, "B_sigma": B_sigma}


class NIWPrior(ImpulsoModel):
    """Natural-conjugate Normal-Inverse-Wishart Minnesota prior (Giannone-Lenza-Primiceri, 2015).

    Distinct from :class:`MinnesotaPrior`: that prior uses an independent-Normal
    coefficient prior with a separate covariance and is sampled with MCMC, whereas this
    prior is conjugate, so the posterior and marginal likelihood are closed-form. The
    conjugate (Kronecker) structure is what buys the closed form; its cost is that
    per-equation own/cross shrinkage asymmetry is not identified (use ``MinnesotaPrior``
    for that).

    Attributes:
        tightness: Overall Minnesota shrinkage ``lambda`` (prior standard deviation).
            Must be > 0. When ``select`` is set this is only the starting value; the
            tightness is estimated by marginal likelihood.
        select: Estimate the tightness from the data (empirical / hierarchical Bayes)
            rather than fixing it.
        decay: Lag-decay exponent on the prior *variance* (GLP ``alpha``); ``2`` gives a
            harmonic decay of the prior standard deviation.
        cross_shrinkage: Shared lag-variance scale; ``1.0`` reproduces GLP (2015). In the
            conjugate prior this is not separately identified from ``tightness``.
        sum_of_coefficients: Sum-of-coefficients prior scale, or ``None`` to disable.
        single_unit_root: Single-unit-root (dummy-initial-observation) prior scale, or
            ``None`` to disable.
        lambda_mode: Mode of the Gamma hyperprior on the tightness (used when ``select``).
        lambda_sd: Standard deviation of the Gamma hyperprior on the tightness.
    """

    tightness: float = Field(0.2, gt=0)
    select: bool = False
    decay: float = Field(2.0, ge=0)
    cross_shrinkage: float = Field(1.0, gt=0)
    sum_of_coefficients: float | None = Field(None, gt=0)
    single_unit_root: float | None = Field(None, gt=0)
    lambda_mode: float = Field(0.2, gt=0)
    lambda_sd: float = Field(0.4, gt=0)

    def build_dummies(
        self,
        y: np.ndarray,
        n_lags: int,
        sigma: np.ndarray | None = None,
        *,
        tightness: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the Minnesota dummy observations this prior implies.

        Args:
            y: Raw data of shape ``(T_full, n_vars)``.
            n_lags: Number of lags.
            sigma: Per-variable scale (AR(1) residual sd). Computed from ``y`` via
                :func:`impulso._conjugate.ar1_residual_sd` when ``None``.
            tightness: Override for ``lambda`` (used when sweeping the marginal
                likelihood during selection); defaults to :attr:`tightness`.

        Returns:
            Tuple ``(Yd, Xd)`` as returned by :func:`impulso._conjugate.minnesota_dummies`.
        """
        if sigma is None:
            sigma = ar1_residual_sd(y)
        lam = self.tightness if tightness is None else tightness
        return minnesota_dummies(
            y,
            n_lags,
            lam=lam,
            decay=self.decay,
            cross=self.cross_shrinkage,
            sigma=sigma,
            mu_sur=self.single_unit_root,
            mu_soc=self.sum_of_coefficients,
        )
