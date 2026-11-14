"""Feature construction for the four pre-specified feature arms.

C  : covariates only
V  : physiological values with the measurement process concealed
P  : measurement-process features only
VP : values (missing left missing) + process features

Every feature at hour t uses only information recorded up to and including t.
Usage: python3 features.py           (builds features for both sites)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402

J = {v: i for i, v in enumerate(cfg.CLIN)}
LAB_IDX = [J[v] for v in cfg.LABS]
LAB_SET = set(LAB_IDX)

COV_NAMES = ["Age", "Sex", "UnitType", "HospAdmTime", "ICULOS"]
LAST_NAMES = [f"{v}_last" for v in cfg.CLIN]
TRAIL_NAMES = [f"{v}_{s}{w}h" for v in cfg.MONITORED for w in (6, 24)
               for s in ("mean", "min", "max")]
DERIV_NAMES = ["ShockIndex", "PulsePressure", "UreaCreatRatio", "SpO2FiO2", "SIRS"]
VAL_NAMES = LAST_NAMES + TRAIL_NAMES + DERIV_NAMES            # 34 + 36 + 5 = 75
PROC_NAMES = ([f"{v}_measured" for v in cfg.CLIN] + [f"{v}_count" for v in cfg.CLIN]
              + [f"{v}_count24h" for v in cfg.CLIN]
              + [f"{v}_hours_since" for v in cfg.CLIN]
              + ["nlab_now", "nlab_6h", "nlab_24h"])           # 139
N_LAST, N_TRAIL = len(LAST_NAMES), len(TRAIL_NAMES)
# column of VAL holding trailing statistics -> index of the clinical variable
TRAIL_VAR = [J[v] for v in cfg.MONITORED for _ in range(6)]


def arm_feature_names(arm):
    return {"C": COV_NAMES, "V": COV_NAMES + VAL_NAMES, "P": COV_NAMES + PROC_NAMES,
            "VP": COV_NAMES + VAL_NAMES + PROC_NAMES}[arm]


def feature_family(name):
    if name in COV_NAMES:
        return "covariate"
    return "process" if name in PROC_NAMES else "value"


def patient_index(pid):
    change = np.r_[True, pid[1:] != pid[:-1]]
    starts = np.flatnonzero(change)
    pidx = (np.cumsum(change) - 1).astype(np.int32)
    row_start = starts[pidx]
    t = (np.arange(len(pid)) - row_start).astype(np.int32)
    return pidx, starts, row_start, t


def derived(L):
    """Indices computed from (possibly concealed) carried-forward values."""
    with np.errstate(divide="ignore", invalid="ignore"):
        hr, sbp, dbp = L[:, J["HR"]], L[:, J["SBP"]], L[:, J["DBP"]]
        si = hr / sbp
        pp = sbp - dbp
        ucr = L[:, J["BUN"]] / L[:, J["Creatinine"]]
        fio2 = L[:, J["FiO2"]]
        fio2 = np.where(fio2 > 1.0, fio2 / 100.0, fio2)   # percentages -> fractions
        fio2 = np.clip(fio2, 0.21, 1.0)
        sf = L[:, J["O2Sat"]] / fio2
        temp, resp = L[:, J["Temp"]], L[:, J["Resp"]]
        paco2, wbc = L[:, J["PaCO2"]], L[:, J["WBC"]]
        sirs = (((temp > 38) | (temp < 36)).astype(np.float32) + (hr > 90)
                + ((resp > 20) | (paco2 < 32)) + ((wbc > 12) | (wbc < 4)))
    out = np.column_stack([si, pp, ucr, sf, sirs]).astype(np.float32)
    out[~np.isfinite(out)] = np.nan
    return out


def trailing(lv, t, w):
    """Trailing-window mean, min and max of a carried-forward series."""
    n = len(lv)
    rows = np.arange(n)
    fin = np.isfinite(lv)
    cs = np.cumsum(np.where(fin, lv, 0.0), dtype=np.float64)
    ck = np.cumsum(fin, dtype=np.int64)
    lo = rows - np.minimum(t, w - 1) - 1          # row just before the window
    safe = np.maximum(lo, 0)
    s = cs - np.where(lo >= 0, cs[safe], 0.0)
    k = ck - np.where(lo >= 0, ck[safe], 0)
    mean = np.where(k > 0, s / np.maximum(k, 1), np.nan).astype(np.float32)
    mn, mx = lv.copy(), lv.copy()
    for d in range(1, w):
        sh = np.full(n, np.nan, np.float32)
        sh[d:] = lv[:-d]
        sh[t < d] = np.nan                         # do not cross patients
        mn = np.fmin(mn, sh)
        mx = np.fmax(mx, sh)
    return mean, mn, mx


def build_features(df):
    pid = df["pid"].to_numpy()
    n = len(df)
    pidx, starts, row_start, t = patient_index(pid)
    rows = np.arange(n)
    X = df[cfg.CLIN].to_numpy(np.float32)
    M = np.isfinite(X)

    unit = np.zeros(n, np.float32)
    unit[df["Unit1"].to_numpy() == 1] = 1          # medical ICU
    unit[df["Unit2"].to_numpy() == 1] = 2          # surgical ICU
    COV = np.column_stack([df["Age"], df["Gender"], unit, df["HospAdmTime"],
                           df["ICULOS"]]).astype(np.float32)

    LAST = np.empty((n, cfg.N_CLIN), np.float32)
    P_cnt = np.empty((n, cfg.N_CLIN), np.float32)
    P_c24 = np.empty((n, cfg.N_CLIN), np.float32)
    P_tsl = np.empty((n, cfg.N_CLIN), np.float32)
    nlab6 = np.zeros(n, np.float32)
    nlab24 = np.zeros(n, np.float32)
    ge6, ge24 = t >= 6, t >= 24
    for j in range(cfg.N_CLIN):
        m = M[:, j]
        c = np.cumsum(m, dtype=np.int32)
        cnt = c - (c[row_start] - m[row_start])     # count so far, this patient
        c24 = cnt.copy()
        c24[ge24] -= cnt[rows[ge24] - 24]
        last = np.maximum.accumulate(np.where(m, rows, -1))
        never = last < row_start
        tsl = (rows - last).astype(np.float32)
        tsl[never] = np.nan
        lv = X[np.maximum(last, 0), j]
        lv[never] = np.nan
        LAST[:, j], P_cnt[:, j], P_c24[:, j], P_tsl[:, j] = lv, cnt, c24, tsl
        if j in LAB_SET:
            c6 = cnt.copy()
            c6[ge6] -= cnt[rows[ge6] - 6]
            nlab6 += c6 > 0
            nlab24 += c24 > 0
    nlab_now = M[:, LAB_IDX].sum(1)
    PROC = np.column_stack([M.astype(np.float32), P_cnt, P_c24, P_tsl,
                            nlab_now, nlab6, nlab24]).astype(np.float32)
    TR = []
    for v in cfg.MONITORED:
        for w in (6, 24):
            TR.extend(trailing(LAST[:, J[v]], t, w))
    VAL = np.column_stack([LAST, np.column_stack(TR), derived(LAST)]).astype(np.float32)
    assert VAL.shape[1] == len(VAL_NAMES) and PROC.shape[1] == len(PROC_NAMES)
    return dict(COV=COV, VAL=VAL, PROC=PROC, X=X, pidx=pidx, t=t, starts=starts)


def make_pools(X, row_mask):
    """Observed raw values per variable among the given (training) rows."""
    Xs = X[row_mask]
    return [Xs[:, j][np.isfinite(Xs[:, j])] for j in range(cfg.N_CLIN)]


def conceal(VAL, pidx, n_pat, pools, rng, mode="hotdeck"):
    """Hide measurement status in the value features.

    hotdeck : never-measured values get a random draw from the training
              distribution, one draw per patient and variable (primary).
    median  : never-measured values get the training median (sensitivity).
    """
    out = np.array(VAL, dtype=np.float32, copy=True)
    if mode == "hotdeck":
        draws = np.column_stack([
            rng.choice(p, size=n_pat) if len(p) else np.zeros(n_pat, np.float32)
            for p in pools]).astype(np.float32)
    else:
        med = np.array([np.median(p) if len(p) else 0.0 for p in pools], np.float32)
        draws = np.broadcast_to(med, (n_pat, cfg.N_CLIN))
    L = out[:, :N_LAST]
    for j in range(cfg.N_CLIN):
        col = L[:, j]
        miss = np.isnan(col)
        col[miss] = draws[pidx[miss], j]
    for k in range(N_TRAIL):
        col = out[:, N_LAST + k]
        miss = np.isnan(col)
        col[miss] = draws[pidx[miss], TRAIL_VAR[k]]
    out[:, N_LAST + N_TRAIL:] = derived(L)
    np.nan_to_num(out, copy=False, nan=0.0)   # derived indices undefined by 0/0
    return out


def patient_table(df, feats):
    g = df.groupby("pid", sort=True)
    y = df["SepsisLabel"].to_numpy()
    pidx, t = feats["pidx"], feats["t"]
    n_pat = int(pidx.max()) + 1
    first_pos = np.full(n_pat, -1, np.int32)
    pos_rows = np.flatnonzero(y == 1)
    first = np.full(n_pat, np.iinfo(np.int32).max, np.int64)
    np.minimum.at(first, pidx[pos_rows], t[pos_rows])
    has = first < np.iinfo(np.int32).max
    first_pos[has] = first[has]
    unit = feats["COV"][feats["starts"], 2]
    # unit type can be recorded later in the stay: take the patient maximum
    unit_max = np.zeros(n_pat, np.float32)
    np.maximum.at(unit_max, pidx, feats["COV"][:, 2])
    tab = pd.DataFrame(dict(
        pid=g["pid"].first().to_numpy(), n_rows=g.size().to_numpy(),
        septic=has.astype(np.int8), first_pos=first_pos,
        age=g["Age"].first().to_numpy(), sex=g["Gender"].first().to_numpy(),
        unit=unit_max, hosp_adm_time=g["HospAdmTime"].first().to_numpy(),
        iculos_first=g["ICULOS"].first().to_numpy(),
        iculos_last=g["ICULOS"].last().to_numpy()))
    skf = StratifiedKFold(cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    fold = np.empty(n_pat, np.int8)
    for k, (_, te) in enumerate(skf.split(np.zeros(n_pat), tab["septic"])):
        fold[te] = k
    tab["fold"] = fold
    del unit
    return tab


def main():
    summary = {}
    for site in cfg.SITES:
        df = pd.read_parquet(cfg.DATA / f"site_{site}.parquet")
        feats = build_features(df)
        for k in ("COV", "VAL", "PROC", "X", "pidx", "t"):
            np.save(cfg.FEAT / f"{site}_{k}.npy", feats[k])
        np.save(cfg.FEAT / f"{site}_y.npy", df["SepsisLabel"].to_numpy(np.int8))
        tab = patient_table(df, feats)
        tab.to_parquet(cfg.FEAT / f"{site}_patients.parquet", index=False)
        summary[site] = dict(rows=int(len(df)), patients=int(len(tab)),
                             septic=int(tab["septic"].sum()),
                             n_val=feats["VAL"].shape[1], n_proc=feats["PROC"].shape[1])
        print(site, summary[site])
    (cfg.FEAT / "feature_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
