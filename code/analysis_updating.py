"""Local updating of the externally developed full model.

For each direction, the external model's predictions at the target site are
recalibrated (logistic recalibration: intercept and slope on the logit scale)
using a random sample of n target-site patients and evaluated in the remaining
patients. The alert threshold is re-derived in the same sample. 50 repetitions
per sample size. Writes results/updating.json.
"""
import json
import os
import sys

import numpy as np
from scipy.special import expit, logit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402
import metrics as mt  # noqa: E402
from train import Site  # noqa: E402

SIZES = [100, 250, 500, 1000, 2500, 5000]
REPS = 50


def fit_recal(lp, y, iters=50):
    """Two-parameter logistic regression of y on lp (Newton-Raphson)."""
    X = np.column_stack([np.ones_like(lp), lp])
    b = np.array([0.0, 1.0])
    for _ in range(iters):
        mu = expit(X @ b)
        g = X.T @ (y - mu)
        H = (X * (mu * (1 - mu))[:, None]).T @ X + 1e-8 * np.eye(2)
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-10:
            break
    return b


def quick_cal(y, s, K=400):
    """O:E, calibration intercept, slope and ICI without bootstrap."""
    edges = np.unique(np.quantile(s, np.linspace(0, 1, K + 1)[1:-1]))
    b = np.searchsorted(edges, s, side="right")
    nb = len(edges) + 1
    pos = np.bincount(b, weights=y, minlength=nb)[None]
    tot = np.bincount(b, minlength=nb).astype(float)
    neg = tot[None] - pos
    safe = np.maximum(tot, 1)
    ms = np.bincount(b, weights=s, minlength=nb) / safe
    lp = np.bincount(b, weights=logit(np.clip(s, mt.EPS, 1 - mt.EPS)), minlength=nb) / safe
    citl = mt.batched_logistic(np.ones((nb, 1)), pos, neg, offset=lp)[0, 0]
    slope = mt.batched_logistic(np.column_stack([np.ones(nb), lp]), pos, neg)[0, 1]
    order = np.argsort(lp)
    cw = np.cumsum(tot[order]) / tot.sum()
    knots = np.interp([0.05, 0.275, 0.5, 0.725, 0.95], cw, lp[order])
    Xs = np.column_stack([np.ones(nb), mt.rcs_basis(lp, knots)])
    fitted = expit(mt.batched_logistic(Xs, pos, neg) @ Xs.T)[0]
    ici = float((tot * np.abs(fitted - ms)).sum() / tot.sum())
    return dict(oe=float(y.sum() / s.sum()), citl=float(citl), slope=float(slope), ici=ici,
                brier=float(np.mean((s - y) ** 2)))


def summarise(rows):
    keys = rows[0].keys()
    return {k: dict(median=float(np.median([r[k] for r in rows])),
                    lo=float(np.percentile([r[k] for r in rows], 2.5)),
                    hi=float(np.percentile([r[k] for r in rows], 97.5))) for k in keys}


def main():
    A, B = Site("A"), Site("B")
    out = {}
    for T, S in ((B, A), (A, B)):
        s = np.load(cfg.PRED / f"ext_{S.name}to{T.name}_VP.npy").astype(float)
        dev = np.load(cfg.PRED / f"int_{S.name}_VP.npy").astype(float)
        lp = logit(np.clip(s, mt.EPS, 1 - mt.EPS))
        tau_dev = mt.AlertEval(S.y, S.pidx, S.t).threshold_for_sensitivity(dev, cfg.SENS_TARGET)
        rng = np.random.default_rng(cfg.SEED + ord(T.name) + 55)
        res = {}
        for n in SIZES:
            before, after, alerts = [], [], []
            for _ in range(REPS):
                pick = np.zeros(T.n_pat, bool)
                pick[rng.choice(T.n_pat, n, replace=False)] = True
                r_fit, r_rest = pick[T.pidx], ~pick[T.pidx]
                a, b = fit_recal(lp[r_fit], T.y[r_fit].astype(float))
                s_new = expit(a + b * lp[r_rest])
                y_rest = T.y[r_rest].astype(float)
                before.append(quick_cal(y_rest, s[r_rest]))
                after.append(dict(quick_cal(y_rest, s_new), fitted_intercept=float(a),
                                  fitted_slope=float(b),
                                  n_septic_in_sample=int(T.septic[pick].sum())))
                # threshold transport: development threshold vs threshold re-derived locally
                new_id = np.cumsum(pick) - 1
                al_fit = mt.AlertEval(T.y[r_fit], new_id[T.pidx[r_fit]], T.t[r_fit])
                rest_id = np.cumsum(~pick) - 1
                al_rest = mt.AlertEval(T.y[r_rest], rest_id[T.pidx[r_rest]], T.t[r_rest])
                if al_fit.septic.sum() >= 5:
                    tau_loc = al_fit.threshold_for_sensitivity(s[r_fit], cfg.SENS_TARGET)
                    e_dev = al_rest.evaluate(s[r_rest], tau_dev)
                    e_loc = al_rest.evaluate(s[r_rest], tau_loc)
                    alerts.append(dict(sens_dev=e_dev["sensitivity"], fa_dev=e_dev["false_alert_prop"],
                                       sens_local=e_loc["sensitivity"], fa_local=e_loc["false_alert_prop"]))
            res[str(n)] = dict(before=summarise(before), after=summarise(after),
                               alerting=summarise(alerts) if alerts else None,
                               ici_improved_prop=float(np.mean([x["ici"] < y_["ici"]
                                                                for x, y_ in zip(after, before)])))
            print(S.name, "->", T.name, n, "ICI before %.4f after %.4f | slope after %.2f (%.2f-%.2f)" % (
                res[str(n)]["before"]["ici"]["median"], res[str(n)]["after"]["ici"]["median"],
                res[str(n)]["after"]["slope"]["median"], res[str(n)]["after"]["slope"]["lo"],
                res[str(n)]["after"]["slope"]["hi"]), flush=True)
        out[f"{S.name}to{T.name}"] = res
    (cfg.RES / "updating.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
