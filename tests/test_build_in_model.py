"""Tests for `VAR.build_in_model` (issue 08a).

`VAR._build_pymc_model` becomes a thin wrapper: it opens a fresh
`pymc.Model`, converts a `VARData` into arrays, and delegates to a new
public `VAR.build_in_model`, which registers a VAR into whichever PyMC
model is active on entry. Two kinds of test live here:

* Parity tests (`TestWrapperLogpParity`, `TestLagDesignMatrixUsage`) exercise
  only the pre-existing `_build_pymc_model` surface and must pass unchanged
  whether the graph is built inline or delegated to `build_in_model` — they
  are not `xfail`-marked.
* Tests of the new public method (`TestBuildInModel`) necessarily fail
  before `build_in_model` exists and are `xfail(strict=True)` until issue
  08a's implementation lands.
"""

import numpy as np
import pandas as pd
import pytest

from impulso.data import VARData
from impulso.spec import VAR


def _make_data(
    rng: np.random.Generator,
    n_vars: int = 2,
    T: int = 60,
    exog_names: list[str] | None = None,
) -> VARData:
    """A small VARData instance, deterministic given `rng`."""
    endog = rng.standard_normal((T, n_vars))
    index = pd.date_range("2000-01-01", periods=T, freq="QS")
    endog_names = [f"y{i + 1}" for i in range(n_vars)]
    if exog_names:
        exog = rng.standard_normal((T, len(exog_names)))
        return VARData(endog=endog, endog_names=endog_names, exog=exog, exog_names=exog_names, index=index)
    return VARData(endog=endog, endog_names=endog_names, index=index)


def _model_logp(model, seed: int = 0) -> float:
    """Total joint log-probability of `model` at a fixed, reproducible point.

    `initial_point(random_seed=...)` is deterministic given the model's
    graph and the seed, so this is a fixed parameter point in the sense
    the acceptance criteria mean, not a random draw that happens to be
    seeded.
    """
    point = model.initial_point(random_seed=seed)
    return float(model.compile_logp()(point))


class TestWrapperLogpParity:
    """Pins `_build_pymc_model`'s numeric output across the 08a refactor.

    The expected values were captured against the pre-refactor
    `_build_pymc_model` (which built the whole graph inline). If
    `build_in_model` changes what graph gets built — a different prior, a
    dropped node, a reordered coordinate — these numbers move and the test
    fails; a pure delegation leaves them untouched. Deliberately independent
    of `build_in_model`: this test exercises only the public `VAR.fit` /
    `_build_pymc_model` surface, so it is not marked `xfail` and must pass
    both before and after the refactor.
    """

    def test_gaussian_no_exog(self, rng):
        data = _make_data(rng)
        model, _ = VAR(lags=1)._build_pymc_model(data)
        assert _model_logp(model) == pytest.approx(-225.00961968371863)

    def test_gaussian_with_exog(self, rng):
        data = _make_data(rng, exog_names=["z"])
        model, _ = VAR(lags=1)._build_pymc_model(data)
        assert _model_logp(model) == pytest.approx(-235.5508133506178)

    def test_student_t_no_exog(self, rng):
        data = _make_data(rng)
        model, _ = VAR(lags=1, error_dist="student_t")._build_pymc_model(data)
        assert _model_logp(model) == pytest.approx(-226.43783005847177)

    def test_student_t_with_exog(self, rng):
        data = _make_data(rng, exog_names=["z"])
        model, _ = VAR(lags=1, error_dist="student_t")._build_pymc_model(data)
        assert _model_logp(model) == pytest.approx(-236.97902372537095)


class TestLagDesignMatrixUsage:
    """`_build_pymc_model` (and, once it exists, `build_in_model`) must use
    the shared `build_lag_design_matrix` from issue 02, not a private
    re-implementation of lag stacking. Not `xfail`-marked: the wrapper
    already routes through the shared builder on `main`.
    """

    def test_wrapper_calls_shared_lag_design_matrix_builder(self, rng, monkeypatch):
        import impulso.spec as spec_module

        data = _make_data(rng)
        calls = []
        original = spec_module.build_lag_design_matrix

        def _spy(endog, n_lags, exog=None):
            calls.append((endog, n_lags, exog))
            return original(endog, n_lags, exog)

        monkeypatch.setattr(spec_module, "build_lag_design_matrix", _spy)

        VAR(lags=1)._build_pymc_model(data)

        assert len(calls) == 1
        got_endog, got_n_lags, got_exog = calls[0]
        np.testing.assert_array_equal(got_endog, data.endog)
        assert got_n_lags == 1
        assert got_exog is None


class TestBuildInModel:
    """Direct tests of the new public `VAR.build_in_model` (issue 08a).

    All `xfail(strict=True)`: `build_in_model` does not exist yet on the
    tests branch. `reason` names the issue so a future maintainer diffing
    xfail reasons for staleness can tell why each one is here.
    """

    @pytest.mark.xfail(strict=True, reason="issue 08a: VAR.build_in_model does not exist yet")
    def test_direct_call_matches_wrapper_logp(self, rng):
        """Calling `build_in_model` directly inside a fresh model gives the
        same log-probability as the wrapper (acceptance criterion 2)."""
        import pymc as pm

        data = _make_data(rng)
        spec = VAR(lags=1)

        wrapper_model, n_lags = spec._build_pymc_model(data)
        wrapper_logp = _model_logp(wrapper_model)

        with pm.Model(coords={"time": data.index[n_lags:]}) as direct_model:
            spec.build_in_model(
                endog=data.endog,
                exog=data.exog,
                n_lags=n_lags,
                endog_names=data.endog_names,
                exog_names=data.exog_names,
            )
        direct_logp = _model_logp(direct_model)

        assert direct_logp == pytest.approx(wrapper_logp)

    @pytest.mark.xfail(strict=True, reason="issue 08a: VAR.build_in_model does not exist yet")
    def test_direct_call_matches_wrapper_logp_with_exog_and_student_t(self, rng):
        import pymc as pm

        data = _make_data(rng, exog_names=["z"])
        spec = VAR(lags=1, error_dist="student_t")

        wrapper_model, n_lags = spec._build_pymc_model(data)
        wrapper_logp = _model_logp(wrapper_model)

        with pm.Model(coords={"time": data.index[n_lags:]}) as direct_model:
            spec.build_in_model(
                endog=data.endog,
                exog=data.exog,
                n_lags=n_lags,
                endog_names=data.endog_names,
                exog_names=data.exog_names,
            )
        direct_logp = _model_logp(direct_model)

        assert direct_logp == pytest.approx(wrapper_logp)

    @pytest.mark.xfail(strict=True, reason="issue 08a: VAR.build_in_model does not exist yet")
    def test_returns_handles_with_intercept_b_bexog_l_and_likelihood(self, rng):
        import pymc as pm

        data = _make_data(rng, exog_names=["z"])
        spec = VAR(lags=1)

        with pm.Model():
            handles = spec.build_in_model(
                endog=data.endog,
                exog=data.exog,
                n_lags=1,
                endog_names=data.endog_names,
                exog_names=data.exog_names,
            )

        assert handles.intercept is not None
        assert handles.B is not None
        assert handles.B_exog is not None
        assert handles.L is not None
        assert handles.obs is not None

    @pytest.mark.xfail(strict=True, reason="issue 08a: VAR.build_in_model does not exist yet")
    def test_returns_none_b_exog_when_no_exog(self, rng):
        import pymc as pm

        data = _make_data(rng)
        spec = VAR(lags=1)

        with pm.Model():
            handles = spec.build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
            )

        assert handles.B_exog is None

    @pytest.mark.xfail(strict=True, reason="issue 08a: VAR.build_in_model does not exist yet")
    def test_registers_into_the_active_model_context(self, rng):
        """No prefix argument: `build_in_model` writes into `pymc.modelcontext(None)`."""
        import pymc as pm

        data = _make_data(rng)
        spec = VAR(lags=1)

        with pm.Model() as model:
            spec.build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
            )

        assert "intercept" in model.named_vars
        assert "B" in model.named_vars
        assert "obs" in model.named_vars

    @pytest.mark.xfail(strict=True, reason="issue 08a: VAR.build_in_model does not exist yet")
    def test_uses_shared_lag_design_matrix_builder(self, rng, monkeypatch):
        import pymc as pm

        import impulso.spec as spec_module

        data = _make_data(rng)
        calls = []
        original = spec_module.build_lag_design_matrix

        def _spy(endog, n_lags, exog=None):
            calls.append((endog, n_lags, exog))
            return original(endog, n_lags, exog)

        monkeypatch.setattr(spec_module, "build_lag_design_matrix", _spy)

        with pm.Model():
            VAR(lags=1).build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
            )

        assert len(calls) == 1
        got_endog, got_n_lags, got_exog = calls[0]
        np.testing.assert_array_equal(got_endog, data.endog)
        assert got_n_lags == 1
        assert got_exog is None

    @pytest.mark.xfail(strict=True, reason="issue 08a: VAR.build_in_model does not exist yet")
    def test_nested_named_model_prefixes_variables_but_not_coords(self, rng):
        """Inside `pm.Model(name=...)`, every Impulso variable, deterministic
        and the likelihood come out prefixed; coords land unprefixed on the
        root model (acceptance criterion 3; PyMC does not prefix coords —
        see `prototype/REPORT.md`)."""
        import pymc as pm

        data = _make_data(rng)
        spec = VAR(lags=1)

        with pm.Model() as root, pm.Model(name="p"):
            spec.build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
            )

        assert "p::intercept" in root.named_vars
        assert "p::B" in root.named_vars
        assert "p::Sigma" in root.named_vars
        assert "p::obs" in root.named_vars
        assert "intercept" not in root.named_vars
        assert "var" in root.coords
        assert list(root.coords["var"]) == data.endog_names

    @pytest.mark.xfail(strict=True, reason="issue 08a: VAR.build_in_model does not exist yet")
    def test_unnested_names_are_unchanged(self, rng):
        """Outside a nested named model, names carry no prefix."""
        import pymc as pm

        data = _make_data(rng)
        spec = VAR(lags=1)

        with pm.Model() as model:
            spec.build_in_model(
                endog=data.endog,
                exog=None,
                n_lags=1,
                endog_names=data.endog_names,
            )

        assert "intercept" in model.named_vars
        assert "B" in model.named_vars
        assert "obs" in model.named_vars

    @pytest.mark.xfail(strict=True, reason="issue 08a: VAR.build_in_model does not exist yet")
    def test_time_coord_length_mismatch_raises(self, rng):
        """Two VARs embedded in the same root model must not silently share
        a stale `time` coordinate when their likelihoods have a different
        number of rows (review round 1). Coords are not prefixed by a
        nested `pm.Model(name=...)`, so the second `build_in_model` call
        would otherwise reuse the first call's `time` coordinate at the
        wrong length, giving the second likelihood a wrong shape that only
        surfaces much later (e.g. in `sample_prior_predictive`)."""
        import pymc as pm

        data_a = _make_data(rng, T=41)  # n_lags=1 -> 40 likelihood rows
        data_b = _make_data(rng, T=31)  # n_lags=1 -> 30 likelihood rows
        spec = VAR(lags=1)

        with pm.Model() as root:
            with pm.Model(name="a"):
                spec.build_in_model(
                    endog=data_a.endog,
                    exog=None,
                    n_lags=1,
                    endog_names=data_a.endog_names,
                )
            with pytest.raises(ValueError, match="time"), pm.Model(name="b"):
                spec.build_in_model(
                    endog=data_b.endog,
                    exog=None,
                    n_lags=1,
                    endog_names=data_b.endog_names,
                )

        # The first VAR's own registration is untouched by the second call's failure.
        assert "a::obs" in root.named_vars

    @pytest.mark.xfail(strict=True, reason="issue 08a: VAR.build_in_model does not exist yet")
    def test_time_coord_matching_length_is_fine(self, rng):
        """Equal-length `time` coords — including the wrapper's real dates —
        are not a collision; only a length mismatch is rejected."""
        import pymc as pm

        data_a = _make_data(rng, T=41)
        data_b = _make_data(rng, T=41)  # same T, different values -- fine
        spec = VAR(lags=1)

        with pm.Model() as root:
            with pm.Model(name="a"):
                spec.build_in_model(
                    endog=data_a.endog,
                    exog=None,
                    n_lags=1,
                    endog_names=data_a.endog_names,
                )
            with pm.Model(name="b"):
                spec.build_in_model(
                    endog=data_b.endog,
                    exog=None,
                    n_lags=1,
                    endog_names=data_b.endog_names,
                )

        assert "a::obs" in root.named_vars
        assert "b::obs" in root.named_vars

    @pytest.mark.xfail(strict=True, reason="issue 08a: VAR.build_in_model does not exist yet")
    def test_endog_scales_overrides_the_default_ar1_residual_sd(self, rng):
        """`endog_scales=None` computes sigma from the data; a caller-supplied
        array is used as-is instead (issue 08a's `endog_scales` argument)."""
        import pymc as pm

        from impulso.spec import _exog_prior_sigma

        data = _make_data(rng, exog_names=["z"])
        spec = VAR(lags=1)
        custom_scales = np.array([2.5, 7.0])

        with pm.Model():
            handles_default = spec.build_in_model(
                endog=data.endog,
                exog=data.exog,
                n_lags=1,
                endog_names=data.endog_names,
                exog_names=data.exog_names,
            )
        with pm.Model():
            handles_custom = spec.build_in_model(
                endog=data.endog,
                exog=data.exog,
                n_lags=1,
                endog_names=data.endog_names,
                exog_names=data.exog_names,
                endog_scales=custom_scales,
            )

        default_sigma = handles_default.B_exog.owner.op.dist_params(handles_default.B_exog.owner)[1].eval()
        custom_sigma = handles_custom.B_exog.owner.op.dist_params(handles_custom.B_exog.owner)[1].eval()

        expected_custom = _exog_prior_sigma(data.endog, data.exog[1:], spec.exog_prior_scale, sigma=custom_scales)
        np.testing.assert_allclose(custom_sigma, expected_custom)
        assert not np.allclose(default_sigma, custom_sigma)
