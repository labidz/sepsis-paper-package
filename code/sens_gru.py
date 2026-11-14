"""Sensitivity analysis: a recurrent neural network (GRU) on the raw hourly
sequences as a deep-learning comparator for the full information set.

Run inside the TensorFlow virtual environment:
    python sens_gru.py   (TensorFlow environment; see requirements-gru.txt)

Design: at each site, cross-validation fold 0 is the test fold. A GRU is
trained on the remaining patients (80% training / 20% early-stopping holdout)
and predicts (i) the test fold of its own site and (ii) the test fold of the
other site. Inputs per hour, for each of the 34 clinical variables: last value
carried forward (standardised, 0 if never measured), measured-this-hour flag,
and scaled time since last measurement; plus the five covariates.
Predictions are written for fold-0 patients only (NaN elsewhere).
"""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

ROOT = os.environ.get("SEPSIS_ROOT", str(Path(__file__).resolve().parents[1] / "work"))
FEAT, PRED, RES = f"{ROOT}/features", f"{ROOT}/preds", f"{ROOT}/results"
SEED, N_CLIN, CAP_H, BATCH, HIDDEN, MAX_EPOCHS, PATIENCE = 20261111, 34, 168.0, 64, 64, 40, 5


class SiteData:
    def __init__(self, name):
        self.name = name
        cov = np.load(f"{FEAT}/{name}_COV.npy")
        val = np.load(f"{FEAT}/{name}_VAL.npy", mmap_mode="r")[:, :N_CLIN]
        proc = np.load(f"{FEAT}/{name}_PROC.npy", mmap_mode="r")
        self.last = np.asarray(val, dtype=np.float32)
        self.meas = np.asarray(proc[:, :N_CLIN], dtype=np.float32)
        tsl = np.asarray(proc[:, 3 * N_CLIN:4 * N_CLIN], dtype=np.float32)
        self.delta = np.where(np.isnan(tsl), 1.0, np.log1p(np.minimum(tsl, CAP_H)) / np.log1p(CAP_H)).astype(np.float32)
        unit = cov[:, 2]
        hosp = np.nan_to_num(cov[:, 3])
        self.cov = np.column_stack([cov[:, 0], cov[:, 1], unit == 1, unit == 2,
                                    np.sign(hosp) * np.log1p(np.abs(hosp)), np.log1p(cov[:, 4])]).astype(np.float32)
        self.y = np.load(f"{FEAT}/{name}_y.npy").astype(np.float32)
        self.pidx = np.load(f"{FEAT}/{name}_pidx.npy")
        self.pat = pd.read_parquet(f"{FEAT}/{name}_patients.parquet")
        self.starts = np.r_[0, np.cumsum(self.pat["n_rows"].to_numpy())]

    def tensor(self, stats):
        mu, sd, cmu, csd = stats
        v = (self.last - mu) / sd
        v[~np.isfinite(v)] = 0.0
        v = np.clip(v, -6, 6)
        c = (self.cov - cmu) / csd
        return np.concatenate([v, self.meas, self.delta, c], axis=1).astype(np.float32)


def fit_stats(d, rows):
    mu = np.nanmean(d.last[rows], axis=0)
    sd = np.nanstd(d.last[rows], axis=0)
    mu, sd = np.nan_to_num(mu), np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    cmu, csd = d.cov[rows].mean(0), d.cov[rows].std(0)
    csd[csd < 1e-6] = 1.0
    cmu[[1, 2, 3]], csd[[1, 2, 3]] = 0.0, 1.0          # leave binary covariates as 0/1
    return mu, sd, cmu, csd


def batches(X, y, starts, pats, shuffle, rng=None):
    order = sorted(pats, key=lambda p: starts[p + 1] - starts[p])
    groups = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
    if shuffle:
        rng.shuffle(groups)
    for g in groups:
        L = max(starts[p + 1] - starts[p] for p in g)
        xb = np.zeros((len(g), L, X.shape[1]), np.float32)
        yb = np.zeros((len(g), L, 1), np.float32)
        wb = np.zeros((len(g), L), np.float32)
        for i, p in enumerate(g):
            a, b = starts[p], starts[p + 1]
            xb[i, :b - a], yb[i, :b - a, 0], wb[i, :b - a] = X[a:b], y[a:b], 1.0
        yield g, xb, yb, wb


def build(n_in):
    inp = tf.keras.Input(shape=(None, n_in))
    h = tf.keras.layers.GRU(HIDDEN, return_sequences=True)(inp)
    h = tf.keras.layers.Dropout(0.2)(h)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(h)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(5e-4, clipnorm=1.0), loss=tf.keras.losses.BinaryCrossentropy(),
              weighted_metrics=[])
    return m


def predict(model, X, d, pats):
    out = np.full(len(d.y), np.nan, np.float32)
    for g, xb, _, wb in batches(X, d.y, d.starts, list(pats), False):
        p = model.predict_on_batch(xb)[:, :, 0]
        for i, pt in enumerate(g):
            a, b = d.starts[pt], d.starts[pt + 1]
            out[a:b] = p[i, :b - a]
    return out


def main():
    tf.keras.utils.set_random_seed(SEED)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    data = {s: SiteData(s) for s in "AB"}
    log = {}
    for s, o in (("A", "B"), ("B", "A")):
        d, e = data[s], data[o]
        fold, septic = d.pat["fold"].to_numpy(), d.pat["septic"].to_numpy()
        dev = np.flatnonzero(fold != 0)
        tr, va = next(StratifiedShuffleSplit(1, test_size=0.2, random_state=SEED + 1).split(dev, septic[dev]))
        tr, va = dev[tr], dev[va]
        rows_tr = np.flatnonzero(np.isin(d.pidx, tr))
        stats = fit_stats(d, rows_tr)
        X, Xe = d.tensor(stats), e.tensor(stats)
        model = build(X.shape[1])
        rng = np.random.default_rng(SEED)
        best, best_w, bad, hist = np.inf, None, 0, []
        t0 = time.time()
        for ep in range(MAX_EPOCHS):
            for _, xb, yb, wb in batches(X, d.y, d.starts, list(tr), True, rng):
                model.train_on_batch(xb, yb, sample_weight=wb)
            num = den = 0.0
            for _, xb, yb, wb in batches(X, d.y, d.starts, list(va), False):
                p = np.clip(model.predict_on_batch(xb)[:, :, 0], 1e-7, 1 - 1e-7)
                ll = -(yb[:, :, 0] * np.log(p) + (1 - yb[:, :, 0]) * np.log(1 - p))
                num += float((ll * wb).sum()); den += float(wb.sum())
            vl = num / den
            hist.append(vl)
            print(s, "epoch", ep, "val_logloss", round(vl, 5), "secs", round(time.time() - t0), flush=True)
            if vl < best - 1e-5:
                best, best_w, bad = vl, model.get_weights(), 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        model.set_weights(best_w)
        np.save(f"{PRED}/int_{s}_GRU_f0.npy", predict(model, X, d, np.flatnonzero(fold == 0)))
        np.save(f"{PRED}/ext_{s}to{o}_GRU_f0.npy",
                predict(model, Xe, e, np.flatnonzero(e.pat["fold"].to_numpy() == 0)))
        log[s] = dict(val_logloss=hist, best=best, epochs=len(hist), secs=round(time.time() - t0),
                      n_train_patients=int(len(tr)), n_params=int(model.count_params()))
    json.dump(log, open(f"{RES}/gru_log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
