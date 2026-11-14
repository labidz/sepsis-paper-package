"""Additional analyses requested in internal review (no refitting).

(a) H2, H3 and the process share by time window relative to onset.
(b) Paired difference in local AUROC, process-only minus values-only.
(c) Challenge utility of the local model with the threshold chosen without the
    evaluation fold (for fold k, chosen on the out-of-fold predictions of the
    other folds).
(d) Net benefit (per 1,000 patient-hours) of the external full model before and
    after logistic recalibration on n target-site patients, and of the local
    model, on the patients not sampled (50 repetitions).
Writes results/rev_extra.json.
"""
import json
import os
import sys

import numpy as np
from scipy.special import expit, logit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402
import metrics as mt  # noqa: E402
from analysis_main import jsonable, load, pval  # noqa: E402
from analysis_updating import fit_recal  # noqa: E402
from train import Site  # noqa: E402

L, E = "local", "external"
WINDOWS = (("-24 to -13 h", -24, -13), ("-12 to -7 h", -12, -7), ("-6 to -1 h", -6, -1), ("0 to +3 h", 0, 3))
NB_TH = (0.01, 0.02, 0.05)
SIZES = (100, 250, 500, 1000, 2500, 5000)
REPS = 50


def contrasts(ev, preds, rows=None, y=None):
    point, boot = {}, {}
    for k, s in preds.items():
        h = ev.hist(s if rows is None else s[rows])
        point[k] = ev.discrimination(None, h)["auroc"]
        boot[k] = ev._auc(h["POS"], h["NEG"])

    def c(coefs):
        est = sum(w * point[k] for k, w in coefs.items())
        b = sum(w * boot[k] for k, w in coefs.items())
        return dict(est=float(est), ci=mt.pct_ci(b), p=pval(b))
    out = dict(H1=c({(E, "VP"): 1, (L, "VP"): -1}), H2=c({(L, "P"): 1, (L, "C"): -1}),
               H3=c({(L, "P"): 1, (E, "P"): -1, (L, "V"): -1, (E, "V"): 1}),
               local_P_minus_V=c({(L, "P"): 1, (L, "V"): -1}),
               external_P_minus_V=c({(E, "P"): 1, (E, "V"): -1}))
    for sc in (L, E):
        out[f"shapley_process_{sc}"] = c({(sc, "P"): .5, (sc, "C"): -.5, (sc, "VP"): .5, (sc, "V"): -.5})
        out[f"shapley_values_{sc}"] = c({(sc, "V"): .5, (sc, "C"): -.5, (sc, "VP"): .5, (sc, "P"): -.5})
        num = .5 * (boot[(sc, "P")] - boot[(sc, "C")] + boot[(sc, "VP")] - boot[(sc, "V")])
        den = boot[(sc, "VP")] - boot[(sc, "C")]
        pnum = .5 * (point[(sc, "P")] - point[(sc, "C")] + point[(sc, "VP")] - point[(sc, "V")])
        out[f"process_share_{sc}"] = dict(est=float(pnum / (point[(sc, "VP")] - point[(sc, "C")])),
                                          ci=mt.pct_ci(num / den))
    return out


def net_benefit(y, s, th):
    pos = s >= th
    n = len(y)
    return 1000 * ((pos & (y == 1)).sum() - (pos & (y == 0)).sum() * th / (1 - th)) / n


def main():
    A, B = Site("A"), Site("B")
    res = {}
    for T, S in ((B, A), (A, B)):
        preds = {(L, a): load(f"int_{T.name}_{a}") for a in cfg.ARMS}
        preds.update({(E, a): load(f"ext_{S.name}to{T.name}_{a}") for a in cfg.ARMS})
        r = {}
        # (a) by window
        al = mt.AlertEval(T.y, T.pidx, T.t)
        septic_row = T.septic[T.pidx] == 1
        d = T.t - al.onset[T.pidx].astype(float)
        r["by_window"] = {}
        for lab, lo, hi in WINDOWS:
            rows = np.flatnonzero(~septic_row | ((d >= lo) & (d <= hi)))
            yy = septic_row[rows].astype(np.int8)
            ev = mt.SiteEval(yy, T.pidx[rows], n_boot=cfg.N_BOOT, seed=cfg.SEED + ord(T.name))
            r["by_window"][lab] = contrasts(ev, preds, rows)
            print(T.name, lab, {k: round(v["est"], 4) for k, v in r["by_window"][lab].items()}, flush=True)
        # (b) primary, paired local P - V and absolute Shapley contributions
        ev = mt.SiteEval(T.y, T.pidx, n_boot=cfg.N_BOOT, seed=cfg.SEED + ord(T.name))
        r["primary"] = contrasts(ev, preds)
        print(T.name, "primary", {k: (round(v["est"], 4), [round(x, 4) for x in v["ci"]]) for k, v in r["primary"].items()}, flush=True)
        # (c) utility with a cross-validated threshold for the local model
        u1, u0, best = mt.utility_weights(T.y, T.pidx, T.t)
        loc = preds[(L, "VP")].astype(float)
        fold_row = T.fold[T.pidx]
        shifted = np.empty_like(loc)
        ths = []
        for k in range(cfg.N_FOLDS):
            other, this = fold_row != k, fold_row == k
            th, _ = mt.best_utility_threshold(loc[other], u1[other], u0[other], best[other])
            shifted[this] = loc[this] - th
            ths.append(th)
        est, ci = mt.utility_ci(shifted, 0.0, u1, u0, best, ev)
        r["utility_local_cv_threshold"] = dict(est=est, ci=ci, thresholds=ths)
        print(T.name, "utility local cv-threshold", round(est, 3), [round(x, 3) for x in ci], flush=True)
        # (d) net benefit after recalibration
        s = preds[(E, "VP")].astype(float)
        lp = logit(np.clip(s, mt.EPS, 1 - mt.EPS))
        rng = np.random.default_rng(cfg.SEED + ord(T.name) + 77)
        nb = {}
        for n in SIZES:
            rec = {th: dict(before=[], after=[], local=[]) for th in NB_TH}
            for _ in range(REPS):
                pick = np.zeros(T.n_pat, bool)
                pick[rng.choice(T.n_pat, n, replace=False)] = True
                r_fit, r_rest = pick[T.pidx], ~pick[T.pidx]
                a, b = fit_recal(lp[r_fit], T.y[r_fit].astype(float))
                s_new = expit(a + b * lp[r_rest])
                y_rest = T.y[r_rest]
                for th in NB_TH:
                    rec[th]["before"].append(net_benefit(y_rest, s[r_rest], th))
                    rec[th]["after"].append(net_benefit(y_rest, s_new, th))
                    rec[th]["local"].append(net_benefit(y_rest, loc[r_rest], th))
            nb[str(n)] = {str(th): {k: dict(median=float(np.median(v)), lo=float(np.quantile(v, .025)),
                                            hi=float(np.quantile(v, .975))) for k, v in rr.items()}
                          for th, rr in rec.items()}
            print(T.name, n, {th: {k: round(v["median"], 2) for k, v in x.items()} for th, x in nb[str(n)].items()}, flush=True)
        r["net_benefit_after_recalibration"] = nb
        res[T.name] = r
    (cfg.RES / "rev_extra.json").write_text(json.dumps(jsonable(res), indent=1))


if __name__ == "__main__":
    main()
