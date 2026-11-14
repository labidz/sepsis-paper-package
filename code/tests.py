"""Correctness tests for features and metrics (run on the synthetic data).

Every vectorised routine is compared with a slow, obviously-correct reference.
Usage: SEPSIS_ROOT=<synthetic work dir> python3 tests.py
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402
import features as ft  # noqa: E402
import metrics as mt  # noqa: E402


def ref_utility(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0,
                max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0):
    """Loop transcribed from the official evaluate_sepsis_score.py."""
    if np.any(labels):
        is_septic, t_sepsis = True, np.argmax(labels) - dt_optimal
    else:
        is_septic, t_sepsis = False, float("inf")
    n = len(labels)
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early); b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal); b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal); b_3 = -m_3 * dt_optimal
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            elif not is_septic and not predictions[t]:
                u[t] = u_tn
    return np.sum(u)


def ref_normalised_utility(y, pred, pidx):
    obs = best = ina = 0.0
    for p in np.unique(pidx):
        lab, pr = y[pidx == p], pred[pidx == p]
        n = len(lab)
        bp = np.zeros(n)
        if np.any(lab):
            ts = int(np.argmax(lab)) + 6
            bp[max(0, ts - 12): min(ts + 3 + 1, n)] = 1
        obs += ref_utility(lab, pr); best += ref_utility(lab, bp)
        ina += ref_utility(lab, np.zeros(n))
    return (obs - ina) / (best - ina)


def test_features(site="A", n_check=40):
    df = pd.read_parquet(cfg.DATA / f"site_{site}.parquet")
    f = ft.build_features(df)
    VAL, PROC, pidx = f["VAL"], f["PROC"], f["pidx"]
    rng = np.random.default_rng(0)
    nP = cfg.N_CLIN
    for p in rng.choice(pidx.max() + 1, n_check, replace=False):
        rows = np.flatnonzero(pidx == p)
        d = df.iloc[rows].reset_index(drop=True)
        X = d[cfg.CLIN].to_numpy(float)
        M = np.isfinite(X)
        L = len(d)
        last = pd.DataFrame(X).ffill().to_numpy()
        assert np.allclose(VAL[rows, :nP], last, equal_nan=True), "last value"
        assert np.array_equal(PROC[rows, :nP], M.astype(np.float32)), "measured"
        assert np.array_equal(PROC[rows, nP:2 * nP], np.cumsum(M, 0)), "count"
        c24 = np.array([[M[max(0, t - 23): t + 1, j].sum() for j in range(nP)] for t in range(L)])
        assert np.array_equal(PROC[rows, 2 * nP:3 * nP], c24), "count24"
        tsl = np.full((L, nP), np.nan)
        for j in range(nP):
            lastt = -1
            for t in range(L):
                if M[t, j]:
                    lastt = t
                if lastt >= 0:
                    tsl[t, j] = t - lastt
        assert np.allclose(PROC[rows, 3 * nP:4 * nP], tsl, equal_nan=True), "hours since"
        lab = ft.LAB_IDX
        now = M[:, lab].sum(1)
        n6 = np.array([M[max(0, t - 5): t + 1][:, lab].any(0).sum() for t in range(L)])
        n24 = np.array([M[max(0, t - 23): t + 1][:, lab].any(0).sum() for t in range(L)])
        assert np.array_equal(PROC[rows, 4 * nP], now) and np.array_equal(PROC[rows, 4 * nP + 1], n6)
        assert np.array_equal(PROC[rows, 4 * nP + 2], n24), "nlab windows"
        k = nP
        for v in cfg.MONITORED:
            s = pd.Series(last[:, ft.J[v]])
            for w in (6, 24):
                r = s.rolling(w, min_periods=1)
                for stat in (r.mean(), r.min(), r.max()):
                    assert np.allclose(VAL[rows, k], stat.to_numpy(), equal_nan=True, atol=1e-3), (v, w)
                    k += 1
    # concealment: no missing values left, observed values untouched
    pools = ft.make_pools(f["X"], slice(None))
    for mode in ("hotdeck", "median"):
        C = ft.conceal(VAL, pidx, int(pidx.max()) + 1, pools, np.random.default_rng(1), mode)
        assert np.isfinite(C).all()
        obs = np.isfinite(VAL[:, :nP])
        assert np.array_equal(C[:, :nP][obs], VAL[:, :nP][obs])
    # hot-deck draws are constant within patient for a never-measured variable
    C = ft.conceal(VAL, pidx, int(pidx.max()) + 1, pools, np.random.default_rng(1))
    j = ft.J["EtCO2"]                      # never measured at synthetic site A
    per = pd.Series(C[:, j]).groupby(pidx).nunique()
    assert (per == 1).all() if site == "A" else True
    print(f"features[{site}]: OK ({n_check} patients checked against loops)")


def test_metrics(site="A"):
    y = np.load(cfg.FEAT / f"{site}_y.npy")
    pidx = np.load(cfg.FEAT / f"{site}_pidx.npy")
    t = np.load(cfg.FEAT / f"{site}_t.npy")
    rng = np.random.default_rng(3)
    s = expit(-3.5 + 2.0 * y + rng.normal(0, 1.2, len(y)))
    # utility
    u1, u0, best = mt.utility_weights(y, pidx, t)
    for th in (0.02, 0.05, 0.2):
        a = mt.normalised_utility(s >= th, u1, u0, best)
        b = ref_normalised_utility(y, (s >= th).astype(int), pidx)
        assert abs(a - b) < 1e-9, (a, b)
    th, val = mt.best_utility_threshold(s, u1, u0, best)
    assert abs(val - mt.normalised_utility(s >= th, u1, u0, best)) < 1e-9
    # discrimination and calibration
    ev = mt.SiteEval(y, pidx, n_boot=300, seed=1)
    h = ev.hist(s)
    d = ev.discrimination(s, h)
    assert abs(d["auroc"] - d["auroc_binned"]) < 5e-4, d
    # brute-force patient bootstrap for AUROC on the first 25 replicates
    order = np.argsort(pidx, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(pidx[order])) + 1, len(y)]
    auc_boot = ev._auc(h["POS"], h["NEG"])
    for b in range(25):
        w = ev.W[b].astype(int)
        idx = np.concatenate([np.tile(order[starts[p]:starts[p + 1]], w[p]) for p in np.flatnonzero(w)])
        assert abs(roc_auc_score(y[idx], s[idx]) - auc_boot[b]) < 2e-3
    c = ev.calibration(s, h)
    lp = logit(np.clip(s, 1e-6, 1 - 1e-6))
    g1 = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
    g2 = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    assert abs(c["citl"] - g1.params[0]) < 5e-3, (c["citl"], g1.params[0])
    assert abs(c["slope"] - g2.params[1]) < 5e-3, (c["slope"], g2.params[1])
    assert abs(c["oe"] - y.sum() / s.sum()) < 1e-9
    assert abs(c["brier"] - np.mean((s - y) ** 2)) < 1e-12
    # a perfectly calibrated score has slope ~1, citl ~0, small ICI
    p_true = expit(-3 + rng.normal(0, 1.0, len(y)))
    y2 = (rng.random(len(y)) < p_true).astype(int)
    ev2 = mt.SiteEval(y2, pidx, n_boot=200, seed=2)
    c2 = ev2.calibration(p_true)
    assert abs(c2["slope"] - 1) < 0.1 and abs(c2["citl"]) < 0.1 and c2["ici"] < 0.01, c2
    c3 = ev2.calibration(expit(1.5 * logit(p_true) + 0.5))      # distorted
    assert c3["slope"] < 0.8 and c3["ici"] > c2["ici"], c3
    # paired difference and net benefit
    s_b = expit(-3.5 + 1.0 * y + rng.normal(0, 1.2, len(y)))
    dd = ev.delta_auc(s, s_b)
    assert dd["delta"] > 0 and dd["ci"][0] > 0
    nb = ev.net_benefit(s, [0.05])[0]
    tp, fp = ((s >= 0.05) & (y == 1)).sum(), ((s >= 0.05) & (y == 0)).sum()
    assert abs(nb["nb"] - (tp - fp * 0.05 / 0.95) / len(y)) < 1e-12
    # alerting against a brute-force loop
    al = mt.AlertEval(y, pidx, t, W=ev.W)
    tau = al.threshold_for_sensitivity(s, 0.8)
    r = al.evaluate(s, tau)
    hit_s = n_s = hit_n = n_n = 0
    leads = []
    for p in np.unique(pidx):
        m = pidx == p
        yy, ss = y[m], s[m]
        if yy.any():
            r1 = int(np.argmax(yy)); onset = r1 + 6
            el = np.arange(len(yy)) <= onset
            n_s += 1
            if (ss[el] >= tau).any():
                hit_s += 1
                if r1 > 0:
                    leads.append(onset - int(np.argmax(ss[el] >= tau)))
        else:
            n_n += 1
            hit_n += (ss >= tau).any()
    assert abs(r["sensitivity"] - hit_s / n_s) < 1e-12 and r["sensitivity"] >= 0.8
    assert abs(r["false_alert_prop"] - hit_n / n_n) < 1e-12
    assert abs(r["ppv"] - hit_s / (hit_s + hit_n)) < 1e-12
    assert abs(r["lead_median"] - np.median(leads)) < 1e-12
    # subgroup evaluator equals evaluating the subset directly
    mask = (np.arange(ev.n_pat) % 2 == 0)
    evs = mt.SiteEval(y, pidx, n_boot=50, seed=1, patient_mask=mask)
    rows = mask[pidx]
    assert abs(evs.discrimination(s)["auroc"] - roc_auc_score(y[rows], s[rows])) < 1e-12
    print(f"metrics[{site}]: OK  (AUROC {d['auroc']:.4f}, binned {d['auroc_binned']:.4f}, "
          f"slope {c['slope']:.3f}, utility threshold {th:.3f})")


if __name__ == "__main__":
    for s in cfg.SITES:
        test_features(s)
    test_metrics("A")
    test_metrics("B")
    print("ALL TESTS PASSED")
