"""Local refitting with limited target-site data (addresses whether local data
can restore discrimination, which logistic recalibration cannot change).

At each target site a fixed random half of the patients is the evaluation set.
From the other half, n patients (n = 500, 1,000, 2,500, 5,000; 5 repetitions)
are sampled and used (i) to fit the full model from scratch and (ii) to continue
boosting the external model (LightGBM init_model), each with early stopping on
a 20% patient-level holdout of the n patients. All models, the unchanged
external model and the cross-validated local model are scored on the same
evaluation patients. Writes results/localrefit.json.
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402
import features as ft  # noqa: E402
from train import Site, fit, inner_split, matrix  # noqa: E402

SIZES = (500, 1000, 2500, 5000)
REPS = 5


def continue_boosting(site, mask, init, seed):
    m_tr, m_va = inner_split(site, mask, seed)
    r_tr, r_va = site.rows_of(m_tr), site.rows_of(m_va)
    names = list(ft.arm_feature_names("VP"))
    dtr = lgb.Dataset(matrix(site, "VP", r_tr), site.y[r_tr], feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(matrix(site, "VP", r_va), site.y[r_va], reference=dtr)
    params = dict(cfg.LGB_PARAMS, seed=seed)
    b = lgb.train(params, dtr, cfg.MAX_ROUNDS, valid_sets=[dva], init_model=init,
                  callbacks=[lgb.early_stopping(cfg.EARLY_STOP, verbose=False)])
    return b, int(b.best_iteration) - init.current_iteration()


def main():
    A, B = Site("A"), Site("B")
    out = {}
    for T, S in ((B, A), (A, B)):
        rng = np.random.default_rng(cfg.SEED + 7)
        perm = rng.permutation(T.n_pat)
        ev_mask = np.zeros(T.n_pat, bool)
        ev_mask[perm[: T.n_pat // 2]] = True
        pool = perm[T.n_pat // 2:]
        rows_ev = T.rows_of(ev_mask)
        y_ev = T.y[rows_ev]
        X_ev = matrix(T, "VP", rows_ev)
        ext = lgb.Booster(model_file=str(cfg.MODELS / f"ext_{S.name}_VP.txt"))
        loc = np.load(cfg.PRED / f"int_{T.name}_VP.npy")[rows_ev]
        res = dict(n_eval_patients=int(ev_mask.sum()), n_eval_septic=int(T.septic[ev_mask].sum()),
                   auroc_external=float(roc_auc_score(y_ev, ext.predict(X_ev))),
                   auroc_local_cv=float(roc_auc_score(y_ev, loc)), sizes={})
        for n in SIZES:
            rec = []
            for r in range(REPS):
                seed = cfg.SEED + 1000 * n + r
                pick = np.random.default_rng(seed).choice(pool, n, replace=False)
                mask = np.zeros(T.n_pat, bool)
                mask[pick] = True
                b_new, _, info = fit([(T, mask)], "VP", seed)
                a_new = roc_auc_score(y_ev, b_new.predict(X_ev, num_iteration=b_new.best_iteration))
                b_cont, added = continue_boosting(T, mask, ext, seed)
                a_cont = roc_auc_score(y_ev, b_cont.predict(X_ev, num_iteration=b_cont.best_iteration))
                rec.append(dict(n_septic=int(T.septic[mask].sum()), auroc_new=float(a_new),
                                auroc_continued=float(a_cont), trees_new=info["best_iter"],
                                trees_added=added))
                print(T.name, n, r, round(a_new, 4), round(a_cont, 4), added, flush=True)
            res["sizes"][str(n)] = rec
        out[T.name] = res
        print(T.name, "external", round(res["auroc_external"], 4), "local", round(res["auroc_local_cv"], 4), flush=True)
    (cfg.RES / "localrefit.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
