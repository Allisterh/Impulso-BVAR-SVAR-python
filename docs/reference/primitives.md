# Primitives

Building blocks shared by Impulso's own estimation and post-processing code,
published for downstream libraries that compose with Impulso posteriors:
the moving-average recursion machinery shared by the IRF, FEVD, and
dynamic-multiplier code (`compute_ma_phi`, `lag_matrices`), and the
per-variable AR(1) residual scale that defines Impulso's data-dependent
Minnesota prior (`ar1_residual_sd`), so external callers can compute the
same scale without reaching into a private module.

```{eval-rst}
.. currentmodule:: impulso

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ar1_residual_sd
   compute_ma_phi
   lag_matrices
```
