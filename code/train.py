"""Model fitting and prediction.

Stages
  external : develop on all patients of one site, predict the other site
  internal : 5-fold patient-grouped cross-validation within each site
  pooled   : 5-fold cross-validation of a model trained on both sites

Usage: python3 train.py <stage> [--arms C,V,P,VP] [--variant NAME]
Every fit writes its predictions to disk and is skipped if they already exist.
"""
import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402
import features as ft  # noqa: E402

DERIV_DEPS = {"ShockIndex": ["HR", "SBP"], "PulsePressure": ["SBP", "DBP"],
              "UreaCreatRatio": ["BUN", "Creatinine"], "SpO2FiO2": ["O2Sat", "FiO2"],
              "SIRS": ["Temp", "HR", "Resp", "PaCO2", "WBC"]}


class Site:
    def __init__(self, name):
        self.name = name
        for k in ("COV", "VAL", "PROC", "X"):
            setattr(self, k, np.load(cfg.FEAT / f"{name}_{k}.npy", mmap_mode="r"))
        self.y = np.load(cfg.FEAT / f"{name}_y.npy")
        self.pidx = np.load(cfg.FEAT / f"{name}_pidx.npy")
        self.t = np.load(cfg.FEAT / f"{name}_t.npy")
        self.pat = pd.read_parquet(cfg.FEAT / f"{name}_patients.parquet")
        self.n_pat = len(self.pat)
        self.septic = self.pat["septic"].to_numpy()
        self.fold = self.pat["fold"].to_numpy()

    def rows_of(self, pat_mask):
        return np.flatnonzero(pat_mask[self.pidx])


def variant_columns(arm, variant, excluded_vars=()):
    """Boolean mask over the arm's features for a sensitivity variant."""
    names = ft.arm_feature_names(arm)
    keep = np.ones(len(names), bool)
    if variant == "strict":                       # current values only
        keep &= ~np.isin(names, ft.TRAIL_NAMES)
    if variant == "harm":                         # harmonised set: lab-count features also removed
        keep &= ~np.isin(names, ["nlab_now", "nlab_6h", "nlab_24h"])
    if variant == "sofa":                         # process features of organ-dysfunction variables removed
        proc = np.isin(names, ft.PROC_NAMES)
        sofa = np.array([n.startswith(tuple(f"{v}_" for v in SOFA_VARS)) for n in names])
        flags = np.array([n.endswith("_measured") for n in names])
        keep &= ~(proc & (sofa | flags)) & ~np.isin(names, ["nlab_now", "nlab_6h", "nlab_24h"])
    if excluded_vars:
        ex = tuple(f"{v}_" for v in excluded_vars)
        drop_der = {d for d, deps in DERIV_DEPS.items() if set(deps) & set(excluded_vars)}
        keep &= np.array([not (n.startswith(ex) or n in drop_der) for n in names])
    return keep


def inner_split(site, train_mask, seed):
    idx = np.flatnonzero(train_mask)
    sss = StratifiedShuffleSplit(1, test_size=cfg.INNER_HOLDOUT, random_state=seed)
    tr, va = next(sss.split(idx, site.septic[idx]))
    m_tr = np.zeros(site.n_pat, bool)
    m_va = np.zeros(site.n_pat, bool)
    m_tr[idx[tr]], m_va[idx[va]] = True, True
    return m_tr, m_va


def matrix(site, arm, rows, pools=None, rng=None, conceal_mode="hotdeck", keep=None):
    if arm == "C":
        X = np.asarray(site.COV[rows])
    elif arm == "P":
        X = np.hstack([site.COV[rows], site.PROC[rows]])
    elif arm == "VP":
        X = np.hstack([site.COV[rows], site.VAL[rows], site.PROC[rows]])
    elif arm == "V":
        v = ft.conceal(site.VAL[rows], site.pidx[rows], site.n_pat, pools, rng, conceal_mode)
        X = np.hstack([site.COV[rows], v])
    else:
        raise ValueError(arm)
    return X if keep is None else X[:, keep]


def fit(site_masks, arm, seed, params=None, conceal_mode="hotdeck", keep=None):
    """site_masks: list of (Site, boolean mask of development patients)."""
    params = dict(cfg.LGB_PARAMS if params is None else params, seed=seed)
    rng = np.random.default_rng(seed)
    splits = []
    for site, mask in site_masks:
        m_tr, m_va = inner_split(site, mask, seed)
        splits.append((site, site.rows_of(m_tr), site.rows_of(m_va)))
    pools = None
    if arm == "V":
        per_site = [ft.make_pools(np.asarray(s.X[r_tr]), slice(None)) for s, r_tr, _ in splits]
        pools = [np.concatenate([p[j] for p in per_site]) for j in range(cfg.N_CLIN)]
    Xtr = np.vstack([matrix(s, arm, r, pools, rng, conceal_mode, keep) for s, r, _ in splits])
    ytr = np.concatenate([s.y[r] for s, r, _ in splits])
    Xva = np.vstack([matrix(s, arm, r, pools, rng, conceal_mode, keep) for s, _, r in splits])
    yva = np.concatenate([s.y[r] for s, _, r in splits])
    names = np.array(ft.arm_feature_names(arm))
    names = list(names if keep is None else names[keep])
    dtr = lgb.Dataset(Xtr, ytr, feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    t0 = time.time()
    booster = lgb.train(params, dtr, cfg.MAX_ROUNDS, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(cfg.EARLY_STOP, verbose=False)])
    info = dict(best_iter=int(booster.best_iteration),
                val_logloss=float(booster.best_score["valid_0"]["binary_logloss"]),
                fit_secs=round(time.time() - t0, 1), n_train_rows=int(len(ytr)),
                n_val_rows=int(len(yva)), n_features=int(Xtr.shape[1]))
    return booster, pools, info


def predict(booster, site, arm, rows, pools, seed, conceal_mode="hotdeck", keep=None,
            chunk=250_000):
    out = np.zeros(len(rows), np.float64)
    n_draw = cfg.N_DRAWS_PRED if (arm == "V" and conceal_mode == "hotdeck") else 1
    for a in range(0, len(rows), chunk):
        r = rows[a:a + chunk]
        for d in range(n_draw):
            rng = np.random.default_rng([seed, d, a])
            X = matrix(site, arm, r, pools, rng, conceal_mode, keep)
            out[a:a + chunk] += booster.predict(X, num_iteration=booster.best_iteration)
    return (out / n_draw).astype(np.float32)


def log(rec):
    with open(cfg.RES / "train_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


SOFA_VARS = ("Platelets", "Bilirubin_total", "Bilirubin_direct", "Creatinine", "FiO2", "pH", "PaCO2", "SaO2", "MAP")
SEED_OFFSET = 0          # refit-variability analysis: --seed-offset N changes every seed


def run_external(src, dst, arm, tag="", **kw):
    out = cfg.PRED / f"ext_{src.name}to{dst.name}_{arm}{tag}.npy"
    if out.exists():
        return
    seed = cfg.SEED + 100 + 1000 * SEED_OFFSET
    booster, pools, info = fit([(src, np.ones(src.n_pat, bool))], arm, seed, **kw)
    booster.save_model(str(cfg.MODELS / f"ext_{src.name}_{arm}{tag}.txt"),
                       num_iteration=booster.best_iteration)
    pk = {k: kw[k] for k in ("conceal_mode", "keep") if k in kw}
    np.save(out, predict(booster, dst, arm, np.arange(len(dst.y)), pools, seed, **pk))
    log(dict(stage="external", src=src.name, dst=dst.name, arm=arm, tag=tag, **info))


def run_internal(site, arm, tag="", **kw):
    out = cfg.PRED / f"int_{site.name}_{arm}{tag}.npy"
    if out.exists():
        return
    oof = np.full(len(site.y), np.nan, np.float32)
    pk = {k: kw[k] for k in ("conceal_mode", "keep") if k in kw}
    for k in range(cfg.N_FOLDS):
        seed = cfg.SEED + 10 * k + 1 + 1000 * SEED_OFFSET
        booster, pools, info = fit([(site, site.fold != k)], arm, seed, **kw)
        rows = site.rows_of(site.fold == k)
        oof[rows] = predict(booster, site, arm, rows, pools, seed, **pk)
        log(dict(stage="internal", site=site.name, arm=arm, tag=tag, fold=k, **info))
    assert not np.isnan(oof).any()
    np.save(out, oof)


def run_pooled(sites, arm, tag="", **kw):
    outs = [cfg.PRED / f"pool_{s.name}_{arm}{tag}.npy" for s in sites]
    if all(o.exists() for o in outs):
        return
    oofs = [np.full(len(s.y), np.nan, np.float32) for s in sites]
    for k in range(cfg.N_FOLDS):
        seed = cfg.SEED + 10 * k + 5
        booster, pools, info = fit([(s, s.fold != k) for s in sites], arm, seed, **kw)
        for s, oof in zip(sites, oofs):
            rows = s.rows_of(s.fold == k)
            oof[rows] = predict(booster, s, arm, rows, pools, seed)
        log(dict(stage="pooled", arm=arm, tag=tag, fold=k, **info))
    for o, oof in zip(outs, oofs):
        np.save(o, oof)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["external", "internal", "pooled"])
    ap.add_argument("--arms", default="VP,P,V,C")
    ap.add_argument("--variant", default="")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--seed-offset", type=int, default=0)
    a = ap.parse_args()
    global SEED_OFFSET
    SEED_OFFSET = a.seed_offset
    A, B = Site("A"), Site("B")
    for arm in a.arms.split(","):
        kw, tag = {}, ""
        if a.variant == "median":
            kw, tag = dict(conceal_mode="median"), "_median"
        elif a.variant == "strict":
            kw, tag = dict(keep=variant_columns(arm, "strict")), "_strict"
        elif a.variant == "excl":
            kw = dict(keep=variant_columns(arm, "", tuple(a.exclude.split(","))))
            tag = "_excl"
        elif a.variant == "sofa":
            kw, tag = dict(keep=variant_columns(arm, "sofa")), "_sofa"
        elif a.variant == "harm":
            kw = dict(keep=None if arm == "C" else variant_columns(arm, "harm", tuple(a.exclude.split(","))))
            tag = "_harm"
        if SEED_OFFSET:
            tag += f"_s{SEED_OFFSET}"
        if a.stage == "external":
            run_external(A, B, arm, tag, **kw)
            run_external(B, A, arm, tag, **kw)
        elif a.stage == "internal":
            run_internal(A, arm, tag, **kw)
            run_internal(B, arm, tag, **kw)
        else:
            run_pooled([A, B], arm, tag, **kw)


if __name__ == "__main__":
    main()
