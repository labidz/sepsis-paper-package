"""Sensitivity analyses: recompute the three hypotheses (and AUROCs) under each
alternative specification. Writes results/sensitivity.json."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402
import metrics as mt  # noqa: E402
from analysis_main import jsonable, pval  # noqa: E402
from train import Site  # noqa: E402


def load(name):
    f = cfg.PRED / f"{name}.npy"
    return np.load(f).astype(np.float64) if f.exists() else None


def evaluate(T, S, tags, patient_mask=None):
    """tags: dict arm -> tag suffix used for that arm's predictions."""
    ev = mt.SiteEval(T.y, T.pidx, n_boot=cfg.N_BOOT, seed=cfg.SEED + ord(T.name), patient_mask=patient_mask)
    pt, bt = {}, {}
    for arm, tag in tags.items():
        for sc, nm in (("local", f"int_{T.name}_{arm}{tag}"), ("external", f"ext_{S.name}to{T.name}_{arm}{tag}")):
            s = load(nm)
            if s is None:
                continue
            s = np.nan_to_num(s, nan=0.0) if patient_mask is not None else s
            h = ev.hist(s)
            pt[(sc, arm)] = ev.discrimination(None, h)["auroc"]
            bt[(sc, arm)] = ev._auc(h["POS"], h["NEG"])

    def con(c):
        if any(k not in pt for k in c):
            return None
        b = sum(w * bt[k] for k, w in c.items())
        return dict(est=float(sum(w * pt[k] for k, w in c.items())), ci=mt.pct_ci(b), p=pval(b))

    L, E = "local", "external"
    out = dict(auroc={f"{k[0]}:{k[1]}": float(v) for k, v in pt.items()},
               H1=con({(E, "VP"): 1, (L, "VP"): -1}), H2=con({(L, "P"): 1, (L, "C"): -1}),
               H3=con({(L, "P"): 1, (E, "P"): -1, (L, "V"): -1, (E, "V"): 1}))
    for sc in (L, E):
        if all((sc, a) in pt for a in cfg.ARMS):
            num = .5 * (bt[(sc, "P")] - bt[(sc, "C")] + bt[(sc, "VP")] - bt[(sc, "V")])
            den = bt[(sc, "VP")] - bt[(sc, "C")]
            pnum = .5 * (pt[(sc, "P")] - pt[(sc, "C")] + pt[(sc, "VP")] - pt[(sc, "V")])
            out[f"process_share_{sc}"] = dict(est=float(pnum / (pt[(sc, "VP")] - pt[(sc, "C")])), ci=mt.pct_ci(num / den))
    return out


def main():
    A, B = Site("A"), Site("B")
    specs = {
        "primary": {"C": "", "V": "", "P": "", "VP": ""},
        "values_median_imputation": {"C": "", "V": "_median", "P": "", "VP": ""},
        "values_current_only": {"C": "", "V": "_strict", "P": "", "VP": ""},
        "excluding_EtCO2": {"C": "", "V": "_excl", "P": "_excl", "VP": "_excl"},
        "logistic_regression": {"C": "_lr", "V": "_lr", "P": "_lr", "VP": "_lr"},
        "tuned_hyperparameters": {"VP": "_tuned"},
    }
    out = {}
    for name, tags in specs.items():
        out[name] = {}
        for T, S in ((B, A), (A, B)):
            if load(f"ext_{S.name}to{T.name}_{list(tags)[-1]}{list(tags.values())[-1]}") is None:
                continue
            out[name][T.name] = evaluate(T, S, tags)
            r = out[name][T.name]
            print(name, T.name, {k: round(v, 3) for k, v in r["auroc"].items()},
                  {h: (None if r[h] is None else (round(r[h]["est"], 4), [round(x, 4) for x in r[h]["ci"]])) for h in ("H1", "H2", "H3")},
                  flush=True)
    # recurrent network: fold-0 patients only, compared with the boosted full model on the same patients
    out["recurrent_network_fold0"] = {}
    for T, S in ((B, A), (A, B)):
        gl, ge = load(f"int_{T.name}_GRU_f0"), load(f"ext_{S.name}to{T.name}_GRU_f0")
        if gl is None or ge is None:
            continue
        mask = T.fold == 0
        ev = mt.SiteEval(T.y, T.pidx, n_boot=cfg.N_BOOT, seed=cfg.SEED + 3, patient_mask=mask)
        hs = {"gru_local": ev.hist(np.nan_to_num(gl)), "gru_external": ev.hist(np.nan_to_num(ge)),
              "gbm_local": ev.hist(load(f"int_{T.name}_VP")), "gbm_external": ev.hist(load(f"ext_{S.name}to{T.name}_VP"))}
        d = {k: ev.discrimination(None, h) for k, h in hs.items()}
        out["recurrent_network_fold0"][T.name] = dict(
            n_patients=int(mask.sum()), n_septic=int(T.septic[mask].sum()),
            auroc={k: dict(est=v["auroc"], ci=v["auroc_ci"]) for k, v in d.items()},
            H1_gru=ev.delta_auc(None, None, hs["gru_external"], hs["gru_local"]),
            H1_gbm=ev.delta_auc(None, None, hs["gbm_external"], hs["gbm_local"]),
            gru_minus_gbm_local=ev.delta_auc(None, None, hs["gru_local"], hs["gbm_local"]),
            gru_minus_gbm_external=ev.delta_auc(None, None, hs["gru_external"], hs["gbm_external"]))
        r = out["recurrent_network_fold0"][T.name]
        print("GRU", T.name, {k: round(v["est"], 3) for k, v in r["auroc"].items()},
              "H1 gru", round(r["H1_gru"]["delta"], 4), [round(x, 4) for x in r["H1_gru"]["ci"]],
              "H1 gbm", round(r["H1_gbm"]["delta"], 4), flush=True)
    (cfg.RES / "sensitivity.json").write_text(json.dumps(jsonable(out), indent=1))


if __name__ == "__main__":
    main()
