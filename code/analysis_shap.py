"""SHAP (TreeSHAP) attributions for the two externally validated full models.

For each model the mean absolute SHAP value per feature is computed on a
random sample of 2,000 patients from its development site and from the
external site. Reported: share of total attribution by feature family
(process / value / covariate), top features, and the agreement of global
importance between the two models. Writes results/shap.json.
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402
import features as ft  # noqa: E402
from train import Site, matrix  # noqa: E402

N_PAT = 2000


def main():
    sites = {"A": Site("A"), "B": Site("B")}
    names = ft.arm_feature_names("VP")
    fam = np.array([ft.feature_family(n) for n in names])
    var = np.array([n.rsplit("_", 1)[0] if ft.feature_family(n) != "covariate" else n for n in names])
    rng = np.random.default_rng(cfg.SEED + 77)
    rows = {}
    for k, s in sites.items():
        pick = np.zeros(s.n_pat, bool)
        pick[rng.choice(s.n_pat, N_PAT, replace=False)] = True
        rows[k] = s.rows_of(pick)
    out, imp = {}, {}
    for dev in ("A", "B"):
        booster = lgb.Booster(model_file=str(cfg.MODELS / f"ext_{dev}_VP.txt"))
        for data in ("A", "B"):
            X = matrix(sites[data], "VP", rows[data])
            contrib = booster.predict(X, pred_contrib=True)[:, :-1]
            m = np.abs(contrib).mean(0)
            imp[(dev, data)] = m
            share = {f: float(m[fam == f].sum() / m.sum()) for f in ("process", "value", "covariate")}
            top = np.argsort(-m)[:20]
            out[f"model_{dev}_on_{data}"] = dict(
                n_rows=int(len(rows[data])), family_share=share,
                top20=[dict(feature=names[i], family=fam[i], mean_abs_shap=float(m[i]),
                            share=float(m[i] / m.sum())) for i in top])
            print(f"model {dev} on data {data}:", {k: round(v, 3) for k, v in share.items()},
                  "| top:", [names[i] for i in top[:8]], flush=True)
    for data in ("A", "B"):
        rho = spearmanr(imp[("A", data)], imp[("B", data)]).statistic
        top_a = set(np.argsort(-imp[("A", data)])[:20])
        top_b = set(np.argsort(-imp[("B", data)])[:20])
        out[f"agreement_on_{data}"] = dict(spearman=float(rho), top20_overlap=len(top_a & top_b))
    # per-variable totals (all features derived from one clinical variable)
    for dev in ("A", "B"):
        m = imp[(dev, dev)]
        tot = {}
        for v, f, x in zip(var, fam, m):
            tot.setdefault((v, f), 0.0)
            tot[(v, f)] += float(x)
        out[f"model_{dev}_by_variable"] = sorted(
            [dict(variable=k[0], family=k[1], share=v / m.sum()) for k, v in tot.items()],
            key=lambda d: -d["share"])[:25]
    (cfg.RES / "shap.json").write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k.startswith("agreement")})


if __name__ == "__main__":
    main()
