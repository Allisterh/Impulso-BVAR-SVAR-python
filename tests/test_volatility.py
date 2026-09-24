"""Tests for the volatility-process seam and adapters."""

import numpy as np
import pytest

from impulso._arviz_compat import make_idata
from impulso.protocols import VolatilityProcess
from impulso.samplers import NUTSSampler
from impulso.spec import VAR
from impulso.volatility import Constant


class TestConstantAdapter:
    def test_constant_satisfies_protocol(self):
        adapter = Constant()
        assert isinstance(adapter, VolatilityProcess)

    def test_constant_name(self):
        assert Constant().name == "constant"

    def test_constant_is_time_varying_false(self):
        assert Constant().is_time_varying is False

    def test_is_time_varying_settable_per_instance(self):
        # Plain instance field (not ClassVar), so a subclass or wrapper can
        # override at construction time.
        assert Constant(is_time_varying=True).is_time_varying is True

    def test_constant_is_frozen(self):
        from pydantic import ValidationError

        adapter = Constant()
        with pytest.raises(ValidationError):
            # Deliberately violate the `Literal["constant"]` static type to
            # assert that the frozen Pydantic model raises at runtime.
            adapter.name = "other"


class TestConstantBuildPymcLatent:
    def test_returns_lower_triangular_for_n_vars_3(self):
        import pymc as pm

        adapter = Constant()
        with pm.Model():
            L_tensor = adapter.build_pymc_latent(n_vars=3, T=100)
            L_value = L_tensor.eval()  # uses prior values; deterministic shape

        assert L_value.shape == (3, 3)
        # Strictly lower-triangular plus positive diagonal: upper triangle is zero.
        upper = np.triu(L_value, k=1)
        assert np.allclose(upper, 0.0)

    def test_handles_n_vars_1(self):
        import pymc as pm

        adapter = Constant()
        with pm.Model():
            L_tensor = adapter.build_pymc_latent(n_vars=1, T=50)
            assert L_tensor.eval().shape == (1, 1)

    def test_registers_expected_pymc_vars(self):
        import pymc as pm

        adapter = Constant()
        with pm.Model() as model:
            adapter.build_pymc_latent(n_vars=3, T=100)

        var_names = {v.name for v in model.unobserved_RVs}
        assert "sigma_sd" in var_names
        assert "tril_offdiag" in var_names

    def test_n_vars_1_skips_tril_offdiag(self):
        import pymc as pm

        adapter = Constant()
        with pm.Model() as model:
            adapter.build_pymc_latent(n_vars=1, T=50)

        var_names = {v.name for v in model.unobserved_RVs}
        assert "sigma_sd" in var_names
        assert "tril_offdiag" not in var_names

    def test_registers_L_deterministic(self):
        """L is exposed as a posterior deterministic so cholesky_at can read it
        without re-decomposing Σ on every call."""
        import pymc as pm

        adapter = Constant()
        with pm.Model() as model:
            adapter.build_pymc_latent(n_vars=3, T=100)

        det_names = {v.name for v in model.deterministics}
        assert "L" in det_names

    def test_accepts_and_ignores_data_kwarg(self):
        """Constant accepts the `data` kwarg for Protocol parity with
        stochastic adapters but ignores it (Σ is data-independent)."""
        import pymc as pm

        adapter = Constant()
        fake_resid = np.zeros((50, 3))
        with pm.Model() as model:
            adapter.build_pymc_latent(n_vars=3, T=50, data=fake_resid)

        assert "L" in {v.name for v in model.deterministics}

    def test_n_vars_2_indexing_and_diagonal(self):
        """Smallest non-trivial off-diagonal case.

        Verifies the single off-diagonal sits at ``L[1, 0]`` (lower-triangular
        cell, not upper) and the diagonal carries the ``sigma_sd`` draws
        (not zeros). A regression that swapped the ``(i, j)`` indexing or
        dropped the ``set_subtensor`` for the diagonal would slip past the
        existing ``n_vars=3`` upper-triangular check.
        """
        import pymc as pm

        adapter = Constant()
        with pm.Model() as model:
            L_tensor = adapter.build_pymc_latent(n_vars=2, T=50)
            L_value, sd_value = pm.draw([L_tensor, model["sigma_sd"]], random_seed=42)

        assert L_value.shape == (2, 2)
        assert L_value[0, 1] == 0.0  # upper-triangular cell stays zero
        assert L_value[1, 0] != 0.0  # off-diagonal placed in the lower-triangular cell
        np.testing.assert_array_equal(np.diag(L_value), sd_value)


class TestPosteriorEquivalence:
    """Sanity check: VAR(lags=1) with default volatility="constant" reproduces
    the *shape* and *variable names* of today's posterior.

    Marked slow because it runs MCMC (cores=1, draws=200, tune=200).
    Intent is regression detection on the seam refactor, not statistical
    correctness — the latter is covered by existing test_fitted.py /
    test_identified.py tests, which we also run to confirm.

    The byte-for-byte equivalence assertion below pins ``nuts_sampler="pymc"``
    explicitly so the determinism contract doesn't depend on whether nutpie
    happens to be installed in the runner's environment. Coverage today is
    only ``lags=1, n_vars=2, no exog``; extending the gate across
    ``(lags, n_vars, exog)`` combinations is tracked as future work.
    """

    @pytest.mark.slow
    def test_default_volatility_posterior_shape_unchanged(self, var_data_2v):
        sampler = NUTSSampler(cores=1, chains=2, draws=200, tune=200, random_seed=42, nuts_sampler="pymc")
        fitted = VAR(lags=1).fit(var_data_2v, sampler=sampler)

        posterior_vars = set(fitted.idata.posterior.data_vars)
        # Variables today's pipeline produces. ``L`` is the cached Cholesky
        # factor of Σ — see Constant.cholesky_at.
        expected = {"intercept", "B", "sigma_sd", "tril_offdiag", "L", "Sigma"}
        assert expected <= posterior_vars, f"Missing posterior variables: {expected - posterior_vars}"

        # Shape contracts: (chains, draws, n_vars[, n_vars]).
        n_chains, n_draws = sampler.chains, sampler.draws
        n_vars = var_data_2v.endog.shape[1]
        assert fitted.idata.posterior["intercept"].shape == (n_chains, n_draws, n_vars)
        assert fitted.idata.posterior["B"].shape == (n_chains, n_draws, n_vars, n_vars)
        assert fitted.idata.posterior["Sigma"].shape == (n_chains, n_draws, n_vars, n_vars)
        # Sigma is symmetric per draw (positive-definiteness is enforced upstream
        # by HalfCauchy(sigma_sd) > 0 and the MvNormal likelihood).
        sigma = fitted.idata.posterior["Sigma"].values
        assert np.allclose(sigma, np.swapaxes(sigma, -1, -2)), "Sigma not symmetric"

    @pytest.mark.slow
    def test_explicit_constant_matches_string_default(self, var_data_2v):
        """VAR(lags=1, volatility="constant") and VAR(lags=1, volatility=Constant())
        and VAR(lags=1) all produce identical posteriors given the same seed."""

        def _fit(spec_kwargs):
            sampler = NUTSSampler(cores=1, chains=2, draws=100, tune=100, random_seed=42, nuts_sampler="pymc")
            return VAR(lags=1, **spec_kwargs).fit(var_data_2v, sampler=sampler)

        fit_default = _fit({})
        fit_string = _fit({"volatility": "constant"})
        fit_object = _fit({"volatility": Constant()})

        for name in ["intercept", "B", "L", "Sigma"]:
            np.testing.assert_array_equal(
                fit_default.idata.posterior[name].values,
                fit_string.idata.posterior[name].values,
                err_msg=f"default vs string disagree on {name}",
            )
            np.testing.assert_array_equal(
                fit_default.idata.posterior[name].values,
                fit_object.idata.posterior[name].values,
                err_msg=f"default vs Constant() instance disagree on {name}",
            )


class TestConstantCholeskyAt:
    def test_returns_cholesky_of_sigma(self, synthetic_idata_2v):
        adapter = Constant()
        L = adapter.cholesky_at(synthetic_idata_2v.posterior, t=None)

        sigma = synthetic_idata_2v.posterior["Sigma"].values
        # L @ L.T should reproduce Sigma.
        reconstructed = np.einsum("cdij,cdkj->cdik", L, L)
        np.testing.assert_allclose(reconstructed, sigma, rtol=1e-6)

    def test_t_argument_ignored_for_constant(self, synthetic_idata_2v):
        """For constant volatility, any value of t returns the same L."""
        adapter = Constant()
        L_none = adapter.cholesky_at(synthetic_idata_2v.posterior, t=None)
        L_zero = adapter.cholesky_at(synthetic_idata_2v.posterior, t=0)
        L_arbitrary = adapter.cholesky_at(synthetic_idata_2v.posterior, t=42)
        np.testing.assert_array_equal(L_none, L_zero)
        np.testing.assert_array_equal(L_none, L_arbitrary)

    def test_returns_lower_triangular(self, synthetic_idata_2v):
        adapter = Constant()
        L = adapter.cholesky_at(synthetic_idata_2v.posterior, t=None)
        # Strictly upper-triangular block must be zero.
        upper = np.triu(L, k=1)
        assert np.allclose(upper, 0.0)

    def test_reads_cached_L_not_decomposed_sigma(self):
        """cholesky_at must read posterior["L"] directly, not recompute
        chol(Σ). Pinned by making L and Σ inconsistent in the fixture and
        asserting the returned array equals L exactly."""
        import xarray as xr

        # L is a fixed lower-triangular factor; Σ is something *different*
        # (identity). A correct implementation returns L. A re-decomposing
        # implementation returns chol(I) = I, which would differ.
        L_truth = np.array([[2.0, 0.0], [0.5, 1.5]])
        L_arr = np.broadcast_to(L_truth, (1, 1, 2, 2)).copy()
        sigma_arr = np.broadcast_to(np.eye(2), (1, 1, 2, 2)).copy()
        posterior = xr.Dataset({
            "L": xr.DataArray(L_arr, dims=["chain", "draw", "var1", "var2"]),
            "Sigma": xr.DataArray(sigma_arr, dims=["chain", "draw", "var1", "var2"]),
        })
        idata = make_idata(posterior=posterior)

        out = Constant().cholesky_at(idata.posterior, t=None)
        np.testing.assert_array_equal(out, L_arr)


class TestConstantForecastCholeskyPath:
    def test_returns_broadcast_shape(self, synthetic_idata_2v):
        adapter = Constant()
        rng = np.random.default_rng(0)
        path = adapter.forecast_cholesky_path(synthetic_idata_2v.posterior, steps=5, rng=rng)
        # (chains, draws, steps, n_vars, n_vars)
        assert path.shape == (2, 50, 5, 2, 2)

    def test_constant_across_steps(self, synthetic_idata_2v):
        """For constant volatility, every forecast step has the same L."""
        adapter = Constant()
        rng = np.random.default_rng(0)
        path = adapter.forecast_cholesky_path(synthetic_idata_2v.posterior, steps=10, rng=rng)
        # path[..., 0, :, :] must equal path[..., k, :, :] for all k.
        np.testing.assert_array_equal(path[..., 0, :, :], path[..., 9, :, :])

    def test_rng_unused_for_constant(self, synthetic_idata_2v):
        """Constant is deterministic given the posterior — rng is accepted but unused."""
        adapter = Constant()
        path_a = adapter.forecast_cholesky_path(synthetic_idata_2v.posterior, steps=3, rng=np.random.default_rng(0))
        path_b = adapter.forecast_cholesky_path(synthetic_idata_2v.posterior, steps=3, rng=np.random.default_rng(99999))
        np.testing.assert_array_equal(path_a, path_b)


class TestConstantCholeskyPath:
    def test_returns_broadcast_shape_for_T(self, synthetic_idata_2v):
        adapter = Constant()
        path = adapter.cholesky_path(synthetic_idata_2v.posterior, T=10)
        # (chains, draws, T, n_vars, n_vars)
        assert path.shape == (2, 50, 10, 2, 2)

    def test_constant_across_time(self, synthetic_idata_2v):
        adapter = Constant()
        path = adapter.cholesky_path(synthetic_idata_2v.posterior, T=5)
        np.testing.assert_array_equal(path[..., 0, :, :], path[..., 4, :, :])


# --------------------------------------------------------------------------------
# Issue 06: per-variable innovation-scale prior (`InnovationScalePrior`,
# `Constant.innovation_scale_priors`).
# --------------------------------------------------------------------------------


class TestConstantDefaultLogpUnchanged:
    """Regression: the default (field-omitted) path's log-probability must not
    change once `innovation_scale_priors` exists on `Constant`.

    Only touches API that already exists today (`Constant.build_pymc_latent`
    with no new field), so — unlike the rest of this section — this test is
    NOT xfailed: it must pass identically before and after the issue 06
    implementation lands.
    """

    def test_default_sigma_sd_matches_halfcauchy_logpdf(self):
        import pymc as pm
        from scipy import stats

        with pm.Model() as model:
            Constant(sigma_sd_beta=1.7).build_pymc_latent(n_vars=3, T=10)

        point = np.array([0.3, 1.2, 4.0])
        logp = pm.logp(model["sigma_sd"], point).eval()
        expected = stats.halfcauchy(scale=1.7).logpdf(point)
        np.testing.assert_allclose(logp, expected, rtol=1e-6)


class TestInnovationScalePrior:
    """Standalone `InnovationScalePrior` spec: family, scale, validation, round-trip."""

    @pytest.mark.parametrize("family", ["halfnormal", "exponential", "halfcauchy"])
    def test_construction_each_family(self, family):
        from impulso.volatility import InnovationScalePrior  # ty: ignore[unresolved-import]

        prior = InnovationScalePrior(family=family, scale=1.0)
        assert prior.family == family
        assert prior.scale == 1.0

    @pytest.mark.parametrize("bad_scale", [0.0, -1.0, -0.01])
    def test_non_positive_scale_raises(self, bad_scale):
        from pydantic import ValidationError

        from impulso.volatility import InnovationScalePrior  # ty: ignore[unresolved-import]

        with pytest.raises(ValidationError):
            InnovationScalePrior(family="halfnormal", scale=bad_scale)

    def test_unknown_family_raises(self):
        from pydantic import ValidationError

        from impulso.volatility import InnovationScalePrior  # ty: ignore[unresolved-import]

        with pytest.raises(ValidationError):
            InnovationScalePrior(family="lognormal", scale=1.0)

    def test_is_frozen(self):
        from pydantic import ValidationError

        from impulso.volatility import InnovationScalePrior  # ty: ignore[unresolved-import]

        prior = InnovationScalePrior(family="halfnormal", scale=1.0)
        with pytest.raises(ValidationError):
            prior.scale = 2.0

    def test_model_dump_round_trip(self):
        from impulso.volatility import InnovationScalePrior  # ty: ignore[unresolved-import]

        prior = InnovationScalePrior(family="exponential", scale=0.4)
        assert InnovationScalePrior.model_validate(prior.model_dump()) == prior


class TestConstantInnovationScalePriorsField:
    """`Constant.innovation_scale_priors`: default, construction, round-trip."""

    def test_default_is_none(self):
        assert Constant().innovation_scale_priors is None

    def test_accepts_tuple_of_innovation_scale_priors(self):
        from impulso.volatility import InnovationScalePrior  # ty: ignore[unresolved-import]

        priors = (
            InnovationScalePrior(family="halfnormal", scale=0.1),
            InnovationScalePrior(family="halfcauchy", scale=2.5),
        )
        adapter = Constant(innovation_scale_priors=priors)  # ty: ignore[pydantic-discarded-extra-argument]
        assert adapter.innovation_scale_priors == priors

    def test_round_trips_with_other_fields(self):
        from impulso.volatility import InnovationScalePrior  # ty: ignore[unresolved-import]

        priors = (
            InnovationScalePrior(family="halfnormal", scale=0.1),
            InnovationScalePrior(family="exponential", scale=0.4),
            InnovationScalePrior(family="halfcauchy", scale=3.0),
        )
        adapter = Constant(  # ty: ignore[pydantic-discarded-extra-argument]
            sigma_sd_beta=1.5, tril_offdiag_sigma=0.25, innovation_scale_priors=priors
        )
        restored = Constant.model_validate(adapter.model_dump())
        assert restored == adapter
        assert restored.innovation_scale_priors == priors

    def test_round_trip_when_omitted(self):
        adapter = Constant()
        restored = Constant.model_validate(adapter.model_dump())
        assert restored == adapter
        assert restored.innovation_scale_priors is None

    def test_nested_validation_rejects_bad_scale_on_model_validate(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Constant.model_validate({"innovation_scale_priors": [{"family": "halfnormal", "scale": -1.0}]})


class TestConstantInnovationScalePriorsBuild:
    """`build_pymc_latent` with `innovation_scale_priors` set: mismatched
    lengths, mixed families, and the marginal prior each family produces."""

    def test_mismatched_length_raises_at_build_time_not_construction(self):
        import pymc as pm

        from impulso.volatility import InnovationScalePrior  # ty: ignore[unresolved-import]

        # Constant does not know n_vars at construction, so this must not raise.
        priors = (InnovationScalePrior(family="halfnormal", scale=0.1),)
        adapter = Constant(innovation_scale_priors=priors)  # ty: ignore[pydantic-discarded-extra-argument]

        with pytest.raises(ValueError, match="innovation_scale_priors"), pm.Model():
            adapter.build_pymc_latent(n_vars=3, T=10)

    def test_mixed_families_register_one_named_rv_per_variable(self):
        import pymc as pm

        from impulso.volatility import InnovationScalePrior  # ty: ignore[unresolved-import]

        priors = (
            InnovationScalePrior(family="halfnormal", scale=0.2),
            InnovationScalePrior(family="halfcauchy", scale=2.5),
            InnovationScalePrior(family="exponential", scale=0.4),
        )
        adapter = Constant(innovation_scale_priors=priors)  # ty: ignore[pydantic-discarded-extra-argument]
        with pm.Model() as model:
            L_tensor = adapter.build_pymc_latent(n_vars=3, T=10)

        var_names = {v.name for v in model.unobserved_RVs}
        assert {"sigma_sd_0", "sigma_sd_1", "sigma_sd_2", "tril_offdiag", "L"} <= var_names
        assert "sigma_sd" not in var_names

        L_value = L_tensor.eval()
        assert L_value.shape == (3, 3)
        np.testing.assert_allclose(np.triu(L_value, k=1), 0.0)

    def test_n_vars_1_skips_tril_offdiag(self):
        import pymc as pm

        from impulso.volatility import InnovationScalePrior  # ty: ignore[unresolved-import]

        adapter = Constant(  # ty: ignore[pydantic-discarded-extra-argument]
            innovation_scale_priors=(InnovationScalePrior(family="halfnormal", scale=0.3),)
        )
        with pm.Model() as model:
            adapter.build_pymc_latent(n_vars=1, T=10)

        var_names = {v.name for v in model.unobserved_RVs}
        assert "sigma_sd_0" in var_names
        assert "tril_offdiag" not in var_names

    def test_n_vars_2_diagonal_uses_declared_scales(self):
        """Off-diagonal construction (`tril_offdiag` scaled by `sd[i]`) is unchanged."""
        import pymc as pm

        from impulso.volatility import InnovationScalePrior  # ty: ignore[unresolved-import]

        priors = (
            InnovationScalePrior(family="halfnormal", scale=0.3),
            InnovationScalePrior(family="halfcauchy", scale=1.0),
        )
        adapter = Constant(innovation_scale_priors=priors)  # ty: ignore[pydantic-discarded-extra-argument]
        with pm.Model() as model:
            L_tensor = adapter.build_pymc_latent(n_vars=2, T=10)
            L_value, sd0, sd1 = pm.draw([L_tensor, model["sigma_sd_0"], model["sigma_sd_1"]], random_seed=42)

        assert L_value.shape == (2, 2)
        assert L_value[0, 1] == 0.0  # upper-triangular cell stays zero
        assert L_value[1, 0] != 0.0  # off-diagonal placed in the lower-triangular cell
        np.testing.assert_array_equal(np.diag(L_value), np.array([sd0, sd1]))

    @pytest.mark.parametrize(
        ("family", "scale", "point"),
        [
            ("halfnormal", 0.2, 0.35),
            ("exponential", 0.4, 0.35),
            ("halfcauchy", 2.5, 0.35),
        ],
    )
    def test_each_family_produces_its_declared_marginal_prior(self, family, scale, point):
        """Acceptance criterion: each supported family, with a given scale,
        produces the intended marginal prior on its diagonal element.

        Checked by comparing the registered RV's `pm.logp` to the reference
        `scipy.stats` density for that family at the same scale -- a
        deterministic distributional check, not a sampling-based one.
        """
        import pymc as pm
        from scipy import stats

        from impulso.volatility import InnovationScalePrior  # ty: ignore[unresolved-import]

        reference = {
            "halfnormal": stats.halfnorm(scale=scale),
            "exponential": stats.expon(scale=scale),
            "halfcauchy": stats.halfcauchy(scale=scale),
        }[family]

        adapter = Constant(  # ty: ignore[pydantic-discarded-extra-argument]
            innovation_scale_priors=(InnovationScalePrior(family=family, scale=scale),)
        )
        with pm.Model() as model:
            adapter.build_pymc_latent(n_vars=1, T=10)

        logp = pm.logp(model["sigma_sd_0"], point).eval()
        assert float(logp) == pytest.approx(reference.logpdf(point), rel=1e-6)

    def test_mixed_families_each_diagonal_entry_uses_its_own_declared_family(self):
        """Same check as above, but with three different families in one model
        (mixed families across variables must work)."""
        import pymc as pm
        from scipy import stats

        from impulso.volatility import InnovationScalePrior  # ty: ignore[unresolved-import]

        priors = (
            InnovationScalePrior(family="halfnormal", scale=0.2),
            InnovationScalePrior(family="exponential", scale=0.4),
            InnovationScalePrior(family="halfcauchy", scale=2.5),
        )
        adapter = Constant(innovation_scale_priors=priors)  # ty: ignore[pydantic-discarded-extra-argument]
        with pm.Model() as model:
            adapter.build_pymc_latent(n_vars=3, T=10)

        point = 0.35
        references = [
            stats.halfnorm(scale=0.2),
            stats.expon(scale=0.4),
            stats.halfcauchy(scale=2.5),
        ]
        for i, reference in enumerate(references):
            logp = pm.logp(model[f"sigma_sd_{i}"], point).eval()
            assert float(logp) == pytest.approx(reference.logpdf(point), rel=1e-6)
