"""Sensitivity analysis: L2-penalised logistic regression as the learner for all
four feature arms (same splits, same concealment as the primary analysis).

Counts and times are log-transformed, values winsorised at the 0.5th/99.5th
training percentiles, missing values (arm VP only) set to the training median,
never-measured "hours since" set to the cap with an explicit ever-measured
flag, and everything standardised. Predictions are written with tag _lr.
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402
import features as ft  # noqa: E402
from train import Site, inner_split, matrix  # noqa: E402

CAP_H = 168.0


class Prep:
    def __init__(self, arm):
        self.names = np.array(ft.arm_feature_names(arm))

    def _raw(self, X):
        n = self.names
        cols, X = [], np.array(X, dtype=np.float32, copy=True)
        for j, name in enumerate(n):
            x = X[:, j]
            if name == "UnitType":
                cols += [(x == 1).astype(np.float32), (x == 2).astype(np.float32)]
            elif name == "HospAdmTime":
                cols.append(np.sign(x) * np.log1p(np.abs(x)))
            elif name == "ICULOS" or name.endswith(("_count", "_count24h")) or name.startswith("nlab_"):
                cols.append(np.log1p(x))
            elif name.endswith("_hours_since"):
                never = np.isnan(x)
                cols.append(np.log1p(np.where(never, CAP_H, np.minimum(x, CAP_H))))
                cols.append((~never).astype(np.float32))
            else:
                cols.append(x)
        return np.column_stack(cols).astype(np.float32)

    def fit(self, X):
        R = self._raw(X)
        self.lo, self.hi = np.nanpercentile(R, [0.5, 99.5], axis=0)
        self.med = np.nan_to_num(np.nanmedian(R, axis=0))
        R = self._clean(R)
        self.mu, self.sd = R.mean(0), R.std(0)
        self.sd[self.sd < 1e-6] = 1.0
        return ((R - self.mu) / self.sd).astype(np.float32)

    def _clean(self, R):
        R = np.clip(R, np.nan_to_num(self.lo), np.nan_to_num(self.hi, nan=0.0))
        bad = ~np.isfinite(R)
        if bad.any():
            R[bad] = np.broadcast_to(self.med, R.shape)[bad]
        return R

    def transform(self, X):
        return ((self._clean(self._raw(X)) - self.mu) / self.sd).astype(np.float32)


def fit(site, mask, arm, seed):
    m_tr, _ = inner_split(site, mask, seed)            # same development rows as the boosted model
    rows = site.rows_of(m_tr)
    rng = np.random.default_rng(seed)
    pools = ft.make_pools(np.asarray(site.X[rows]), slice(None)) if arm == "V" else None
    prep = Prep(arm)
    Xtr = prep.fit(matrix(site, arm, rows, pools, rng))
    t0 = time.time()
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200).fit(Xtr, site.y[rows])
    return lr, prep, pools, round(time.time() - t0, 1)


def predict(lr, prep, site, arm, rows, pools, seed, chunk=250_000):
    out = np.zeros(len(rows))
    n_draw = cfg.N_DRAWS_PRED if arm == "V" else 1
    for a in range(0, len(rows), chunk):
        r = rows[a:a + chunk]
        for d in range(n_draw):
            rng = np.random.default_rng([seed, d, a])
            out[a:a + chunk] += lr.predict_proba(prep.transform(matrix(site, arm, r, pools, rng)))[:, 1]
    return (out / n_draw).astype(np.float32)


def main():
    A, B = Site("A"), Site("B")
    for arm in ("VP", "P", "V", "C"):
        for src, dst in ((A, B), (B, A)):
            out = cfg.PRED / f"ext_{src.name}to{dst.name}_{arm}_lr.npy"
            if not out.exists():
                seed = cfg.SEED + 100
                lr, prep, pools, secs = fit(src, np.ones(src.n_pat, bool), arm, seed)
                np.save(out, predict(lr, prep, dst, arm, np.arange(len(dst.y)), pools, seed))
                print(json.dumps(dict(stage="external", src=src.name, arm=arm, secs=secs,
                                      n_iter=int(lr.n_iter_[0]))), flush=True)
        for site in (A, B):
            out = cfg.PRED / f"int_{site.name}_{arm}_lr.npy"
            if out.exists():
                continue
            oof = np.full(len(site.y), np.nan, np.float32)
            for k in range(cfg.N_FOLDS):
                seed = cfg.SEED + 10 * k + 1
                lr, prep, pools, secs = fit(site, site.fold != k, arm, seed)
                rows = site.rows_of(site.fold == k)
                oof[rows] = predict(lr, prep, site, arm, rows, pools, seed)
                print(json.dumps(dict(stage="internal", site=site.name, arm=arm, fold=k, secs=secs)), flush=True)
            np.save(out, oof)


if __name__ == "__main__":
    main()
