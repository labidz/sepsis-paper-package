"""Sensitivity analysis: does hyperparameter search change the conclusions?

For the full model (arm VP) at each development site, eight random
configurations are fitted on the same development/holdout split as the primary
external model. The configuration with the lowest holdout log-loss is then
(i) applied to the other site and (ii) cross-validated locally, giving a
"tuned" version of hypothesis H1. Writes results/tuning.json and predictions
with tag _tuned.
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402
import train as tr  # noqa: E402

N_CONFIG = 8


def sample_configs(rng):
    out = []
    for _ in range(N_CONFIG):
        out.append(dict(cfg.LGB_PARAMS,
                        learning_rate=float(rng.choice([0.02, 0.05, 0.1])),
                        num_leaves=int(rng.choice([15, 31, 63, 127])),
                        min_data_in_leaf=int(rng.choice([50, 200, 1000])),
                        feature_fraction=float(rng.choice([0.3, 0.5, 0.8])),
                        bagging_fraction=float(rng.choice([0.6, 0.8, 1.0])),
                        lambda_l2=float(rng.choice([1.0, 10.0, 100.0]))))
    return out


def main():
    A, B = tr.Site("A"), tr.Site("B")
    rng = np.random.default_rng(cfg.SEED + 404)
    configs = sample_configs(rng)
    keys = ("learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction",
            "bagging_fraction", "lambda_l2")
    out = {}
    for src, dst in ((A, B), (B, A)):
        seed = cfg.SEED + 100
        rows = []
        best = None
        for i, params in enumerate(configs):
            booster, _, info = tr.fit([(src, np.ones(src.n_pat, bool))], "VP", seed, params=params)
            pred = tr.predict(booster, dst, "VP", np.arange(len(dst.y)), None, seed)
            rec = dict({k: params[k] for k in keys}, val_logloss=info["val_logloss"],
                       best_iter=info["best_iter"], external_auroc=float(roc_auc_score(dst.y, pred)))
            rows.append(rec)
            print(src.name, i, json.dumps(rec), flush=True)
            if best is None or info["val_logloss"] < best[0]:
                best = (info["val_logloss"], i, pred)
        np.save(cfg.PRED / f"ext_{src.name}to{dst.name}_VP_tuned.npy", best[2])
        out[src.name] = dict(configs=rows, selected=best[1],
                             fixed_external_auroc=float(roc_auc_score(
                                 dst.y, np.load(cfg.PRED / f"ext_{src.name}to{dst.name}_VP.npy"))))
        # local cross-validation with the selected configuration
        oof = np.full(len(src.y), np.nan, np.float32)
        for k in range(cfg.N_FOLDS):
            s2 = cfg.SEED + 10 * k + 1
            booster, _, _ = tr.fit([(src, src.fold != k)], "VP", s2, params=configs[best[1]])
            r = src.rows_of(src.fold == k)
            oof[r] = tr.predict(booster, src, "VP", r, None, s2)
        np.save(cfg.PRED / f"int_{src.name}_VP_tuned.npy", oof)
        out[src.name]["tuned_local_auroc"] = float(roc_auc_score(src.y, oof))
        out[src.name]["fixed_local_auroc"] = float(roc_auc_score(
            src.y, np.load(cfg.PRED / f"int_{src.name}_VP.npy")))
    (cfg.RES / "tuning.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
