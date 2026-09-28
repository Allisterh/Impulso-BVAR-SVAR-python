"""Tests for lag order selection."""

import numpy as np
import pandas as pd
import pytest

from impulso._lag_selection import select_lag_order
from impulso.data import VARData
from impulso.results import LagOrderResult


@pytest.fixture
def var_data(var_data_3v_dgp2):
    return var_data_3v_dgp2


@pytest.fixture
def var_data_2v_with_exog():
    rng = np.random.default_rng(42)
    T, n = 200, 2
    y = np.zeros((T, n))
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + rng.standard_normal(n) * 0.1
    exog = rng.standard_normal((T, 1))
    index = pd.date_range("2000-01-01", periods=T, freq="QS")
    return VARData(endog=y, endog_names=["y1", "y2"], exog=exog, exog_names=["x1"], index=index)


def _inline_criteria_table(y: np.ndarray, exog: np.ndarray | None, max_lags: int) -> pd.DataFrame:
    T, n = y.shape
    rows = []
    for p in range(1, max_lags + 1):
        T_eff = T - p
        x_lag = np.hstack([y[p - lag : T - lag] for lag in range(1, p + 1)])
        X_parts = [np.ones((T_eff, 1)), x_lag]
        if exog is not None:
            X_parts.append(exog[p:])
        X = np.hstack(X_parts)
        Y = y[p:]

        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid = Y - X @ beta
        sigma = (resid.T @ resid) / T_eff

        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0:
            logdet = np.inf

        k_params = X.shape[1] * n
        aic = logdet + 2 * k_params / T_eff
        bic = logdet + np.log(T_eff) * k_params / T_eff
        hq = logdet + 2 * np.log(np.log(T_eff)) * k_params / T_eff

        rows.append({"lag": p, "aic": aic, "bic": bic, "hq": hq})

    return pd.DataFrame(rows).set_index("lag")


class TestSelectLagOrder:
    def test_returns_lag_order_result(self, var_data):
        result = select_lag_order(var_data, max_lags=8)
        assert isinstance(result, LagOrderResult)

    def test_aic_bic_hq_are_positive_ints(self, var_data):
        result = select_lag_order(var_data, max_lags=8)
        assert result.aic >= 1
        assert result.bic >= 1
        assert result.hq >= 1

    def test_summary_has_expected_columns(self, var_data):
        result = select_lag_order(var_data, max_lags=8)
        summary = result.summary()
        assert "aic" in summary.columns
        assert "bic" in summary.columns
        assert "hq" in summary.columns

    def test_summary_rows_match_max_lags(self, var_data):
        result = select_lag_order(var_data, max_lags=6)
        assert len(result.summary()) == 6


class TestSelectLagOrderGoldenEquality:
    def test_golden_equality_with_inline_computation_2v(self, var_data_2v):
        max_lags = 4
        result = select_lag_order(var_data_2v, max_lags=max_lags)
        expected = _inline_criteria_table(var_data_2v.endog, None, max_lags)

        pd.testing.assert_frame_equal(result.summary(), expected)
        assert result.aic == int(expected["aic"].idxmin())
        assert result.bic == int(expected["bic"].idxmin())
        assert result.hq == int(expected["hq"].idxmin())

    def test_golden_equality_with_inline_computation_3v_dgp2(self, var_data_3v_dgp2):
        max_lags = 5
        result = select_lag_order(var_data_3v_dgp2, max_lags=max_lags)
        expected = _inline_criteria_table(var_data_3v_dgp2.endog, None, max_lags)

        pd.testing.assert_frame_equal(result.summary(), expected)
        assert result.aic == int(expected["aic"].idxmin())
        assert result.bic == int(expected["bic"].idxmin())
        assert result.hq == int(expected["hq"].idxmin())

    def test_golden_equality_with_inline_computation_with_exog(self, var_data_2v_with_exog):
        max_lags = 4
        result = select_lag_order(var_data_2v_with_exog, max_lags=max_lags)
        expected = _inline_criteria_table(var_data_2v_with_exog.endog, var_data_2v_with_exog.exog, max_lags)

        pd.testing.assert_frame_equal(result.summary(), expected)
        assert result.aic == int(expected["aic"].idxmin())
        assert result.bic == int(expected["bic"].idxmin())
        assert result.hq == int(expected["hq"].idxmin())
