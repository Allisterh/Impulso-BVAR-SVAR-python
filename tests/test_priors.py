"""Tests for prior specifications."""

import numpy as np
import pytest
from pydantic import ValidationError

from impulso.priors import MinnesotaPrior
from impulso.protocols import Prior


class TestMinnesotaPrior:
    def test_default_construction(self):
        prior = MinnesotaPrior()
        assert prior.tightness == 0.1
        assert prior.decay == "harmonic"
        assert prior.cross_shrinkage == 0.5

    def test_custom_construction(self):
        prior = MinnesotaPrior(tightness=0.2, decay="geometric", cross_shrinkage=0.8)
        assert prior.tightness == 0.2
        assert prior.decay == "geometric"

    def test_frozen(self):
        prior = MinnesotaPrior()
        with pytest.raises(ValidationError):
            prior.tightness = 0.5

    def test_satisfies_prior_protocol(self):
        prior = MinnesotaPrior()
        assert isinstance(prior, Prior)

    @pytest.mark.parametrize("bad_tightness", [0.0, -0.1, -1.0])
    def test_rejects_invalid_tightness(self, bad_tightness):
        with pytest.raises(ValidationError):
            MinnesotaPrior(tightness=bad_tightness)

    @pytest.mark.xfail(strict=True, reason="issue 07a: build_priors doesn't accept sigma yet")
    def test_build_priors_returns_dict(self):
        prior = MinnesotaPrior()
        result = prior.build_priors(n_vars=3, n_lags=2, sigma=np.ones(3))
        assert isinstance(result, dict)
        assert "B_mu" in result
        assert "B_sigma" in result

    @pytest.mark.xfail(strict=True, reason="issue 07a: sigma isn't a parameter yet, so nothing is missing")
    def test_sigma_is_required(self):
        """`sigma` has no default — a caller must always supply it (issue 07a)."""
        prior = MinnesotaPrior()
        with pytest.raises(TypeError):
            prior.build_priors(n_vars=3, n_lags=2)

    @pytest.mark.xfail(strict=True, reason="issue 07a: build_priors doesn't accept sigma yet, so no length to mismatch")
    @pytest.mark.parametrize("bad_sigma", [np.ones(2), np.ones(4), np.ones((3, 1)), np.array(1.0)])
    def test_rejects_sigma_with_wrong_length(self, bad_sigma):
        """`sigma` must have shape `(n_vars,)` — a mismatch is a caller bug, not silently broadcast."""
        prior = MinnesotaPrior()
        with pytest.raises(ValueError, match="sigma must have shape"):
            prior.build_priors(n_vars=3, n_lags=2, sigma=bad_sigma)


class TestMinnesotaPriorCrossLagScaling:
    """Cross-lag prior sd scales by sigma_i/sigma_j; own lags are unchanged
    (issue 07a). `build_priors` is a pure function of `sigma`, so these
    properties are checked directly against varying `sigma` inputs rather
    than through a fitted model.
    """

    @pytest.mark.xfail(strict=True, reason="issue 07a: build_priors doesn't accept sigma yet")
    def test_own_lags_unchanged_by_sigma(self):
        """Own-lag entries (i == j) carry ratio sigma_i/sigma_i == 1."""
        prior = MinnesotaPrior()
        n_vars, n_lags = 3, 3
        baseline = prior.build_priors(n_vars=n_vars, n_lags=n_lags, sigma=np.ones(n_vars))["B_sigma"]
        heterogeneous = prior.build_priors(n_vars=n_vars, n_lags=n_lags, sigma=np.array([1.0, 4.0, 9.0]))["B_sigma"]

        for i in range(n_vars):
            for lag in range(n_lags):
                col = lag * n_vars + i  # own-variable column at this lag
                assert heterogeneous[i, col] == pytest.approx(baseline[i, col])

    @pytest.mark.xfail(strict=True, reason="issue 07a: build_priors doesn't accept sigma yet")
    def test_cross_lag_scaled_by_sigma_ratio(self):
        """Cross-lag entries scale by exactly sigma_i/sigma_j vs. the sigma=1 baseline."""
        prior = MinnesotaPrior()
        n_vars, n_lags = 3, 2
        sigma = np.array([2.0, 3.0, 7.0])
        baseline = prior.build_priors(n_vars=n_vars, n_lags=n_lags, sigma=np.ones(n_vars))["B_sigma"]
        scaled = prior.build_priors(n_vars=n_vars, n_lags=n_lags, sigma=sigma)["B_sigma"]

        for i in range(n_vars):
            for lag in range(n_lags):
                for j in range(n_vars):
                    col = lag * n_vars + j
                    expected = baseline[i, col] * (sigma[i] / sigma[j])
                    assert scaled[i, col] == pytest.approx(expected)

    @pytest.mark.xfail(strict=True, reason="issue 07a: build_priors doesn't accept sigma yet")
    def test_matches_previous_sd_times_sigma_ratio(self):
        """Direct statement of the acceptance criterion: new sd == old sd * sigma_i/sigma_j."""
        prior = MinnesotaPrior(tightness=0.2, decay="geometric", cross_shrinkage=0.3)
        n_vars, n_lags = 2, 4
        sigma = np.array([1.5, 6.0])

        old_style = prior.build_priors(n_vars=n_vars, n_lags=n_lags, sigma=np.ones(n_vars))["B_sigma"]
        new_style = prior.build_priors(n_vars=n_vars, n_lags=n_lags, sigma=sigma)["B_sigma"]

        ratio = sigma[:, None] / sigma[None, :]  # ratio[i, j] = sigma_i/sigma_j
        col_var = np.arange(n_vars * n_lags) % n_vars
        expected = old_style * ratio[:, col_var]
        np.testing.assert_allclose(new_style, expected)

    @pytest.mark.xfail(strict=True, reason="issue 07a: build_priors doesn't accept sigma yet")
    def test_unit_invariance_rescaling_regressor_variable(self):
        """Rescaling variable j by c scales the prior sd on coefficient (i, j) by 1/c.

        `B_exog[i, j]`-style reasoning applies to lag coefficients too: if
        variable j's units are rescaled by c, the coefficient converting j's
        units into i's must shrink by 1/c to describe the same belief about
        the contribution `beta_ij * y_j`. `ar1_residual_sd` scales linearly
        under a linear rescaling of the series, so sigma_j -> c * sigma_j
        models exactly that.
        """
        prior = MinnesotaPrior()
        n_vars, n_lags = 3, 1
        sigma = np.array([2.0, 3.0, 7.0])
        c = 4.0

        base = prior.build_priors(n_vars=n_vars, n_lags=n_lags, sigma=sigma)["B_sigma"]

        sigma_j_rescaled = sigma.copy()
        sigma_j_rescaled[1] *= c  # rescale variable j=1
        rescaled = prior.build_priors(n_vars=n_vars, n_lags=n_lags, sigma=sigma_j_rescaled)["B_sigma"]

        # Coefficient (i=0, j=1): prior sd shrinks by 1/c.
        assert rescaled[0, 1] == pytest.approx(base[0, 1] / c)
        # Coefficient (i=2, j=1): same story, any equation i != j.
        assert rescaled[2, 1] == pytest.approx(base[2, 1] / c)

    @pytest.mark.xfail(strict=True, reason="issue 07a: build_priors doesn't accept sigma yet")
    def test_unit_invariance_rescaling_dependent_variable(self):
        """Rescaling variable i by c scales the prior sd on coefficient (i, j) by c."""
        prior = MinnesotaPrior()
        n_vars, n_lags = 3, 1
        sigma = np.array([2.0, 3.0, 7.0])
        c = 4.0

        base = prior.build_priors(n_vars=n_vars, n_lags=n_lags, sigma=sigma)["B_sigma"]

        sigma_i_rescaled = sigma.copy()
        sigma_i_rescaled[0] *= c  # rescale equation variable i=0
        rescaled = prior.build_priors(n_vars=n_vars, n_lags=n_lags, sigma=sigma_i_rescaled)["B_sigma"]

        assert rescaled[0, 1] == pytest.approx(base[0, 1] * c)
        assert rescaled[0, 2] == pytest.approx(base[0, 2] * c)
        # Own lag (i=0, j=0) is untouched by either rescaling — ratio stays 1.
        assert rescaled[0, 0] == pytest.approx(base[0, 0])
