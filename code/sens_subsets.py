"""Sensitivity analyses on evaluation subsets (no refitting): all three
hypotheses and the process share, re-estimated (i) excluding patients whose
label was positive at their first recorded hour and (ii) excluding hours after
onset. Uses the primary predictions of all four arms. Writes
results/sens_subsets.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402
import metrics as mt  # noqa: E402
from analysis_main import jsonable, load, pval  # noqa: E402
from train import Site  # noqa: E402

L, E = "local", "external"


def hypotheses(ev, preds, rows=None):
    point, boot = {}, {}
    for k, s in preds.items():
        s = s if rows is None else s[rows]
        h = ev.hist(s)
        point[k] = ev.discrimination(None, h)["auroc"]
        boot[k] = ev._auc(h["POS"], h["NEG"])

    def contrast(coefs):
        est = sum(w * point[k] for k, w in coefs.items())
        b = sum(w * boot[k] for k, w in coefs.items())
        return dict(est=float(est), ci=mt.pct_ci(b), p=pval(b))

    out = dict(
        auroc={f"{k[0]}:{k[1]}": float(v) for k, v in point.items()},
        H1=contrast({(E, "VP"): 1, (L, "VP"): -1}),
        H2=contrast({(L, "P"): 1, (L, "C"): -1}),
        H3=contrast({(L, "P"): 1, (E, "P"): -1, (L, "V"): -1, (E, "V"): 1}))
    for sc in (L, E):
        num_b = .5 * (boot[(sc, "P")] - boot[(sc, "C")] + boot[(sc, "VP")] - boot[(sc, "V")])
        den_b = boot[(sc, "VP")] - boot[(sc, "C")]
        num = .5 * (point[(sc, "P")] - point[(sc, "C")] + point[(sc, "VP")] - point[(sc, "V")])
        den = point[(sc, "VP")] - point[(sc, "C")]
        out[f"process_share_{sc}"] = dict(est=float(num / den), ci=mt.pct_ci(num_b / den_b))
    return out


def main():
    A, B = Site("A"), Site("B")
    res = {}
    for T, S in ((B, A), (A, B)):
        preds = {(L, a): load(f"int_{T.name}_{a}") for a in cfg.ARMS}
        preds.update({(E, a): load(f"ext_{S.name}to{T.name}_{a}") for a in cfg.ARMS})
        al = mt.AlertEval(T.y, T.pidx, T.t)
        keep_pat = T.pat["first_pos"].to_numpy() != 0
        ev1 = mt.SiteEval(T.y, T.pidx, n_boot=cfg.N_BOOT, seed=cfg.SEED + 9, patient_mask=keep_pat)
        r1 = hypotheses(ev1, preds)
        r1.update(n_patients=int(keep_pat.sum()), n_septic=int(T.septic[keep_pat].sum()))
        rows = np.flatnonzero(T.t <= al.onset[T.pidx])
        ev2 = mt.SiteEval(T.y[rows], T.pidx[rows], n_boot=cfg.N_BOOT, seed=cfg.SEED + ord(T.name))
        r2 = hypotheses(ev2, preds, rows)
        r2.update(n_rows=int(len(rows)))
        res[T.name] = dict(excluding_patients_positive_at_first_hour=r1, excluding_hours_after_onset=r2)
        for k, r in res[T.name].items():
            print(T.name, k, {h: (round(r[h]["est"], 4), [round(x, 4) for x in r[h]["ci"]]) for h in ("H1", "H2", "H3")},
                  round(r["process_share_local"]["est"], 3), round(r["process_share_external"]["est"], 3), flush=True)
    (cfg.RES / "sens_subsets.json").write_text(json.dumps(jsonable(res), indent=1))


if __name__ == "__main__":
    main()
