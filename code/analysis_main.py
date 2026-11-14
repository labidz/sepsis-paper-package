"""Primary analysis: discrimination of the four feature arms in local
(internal cross-validation) and external validation, the three pre-specified
hypotheses, calibration, clinical utility, alerting, subgroups and the pooled
model. Writes results/main_<site>.json for each target site.

Usage: python3 analysis_main.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402
import metrics as mt  # noqa: E402
from train import Site  # noqa: E402


def load(name):
    f = cfg.PRED / f"{name}.npy"
    return np.load(f).astype(np.float64) if f.exists() else None


def pval(boot):
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return float(max(p, 1.0 / len(boot)))


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items() if k != "boot"}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def subgroup_masks(site):
    pat = site.pat
    age = np.digitize(pat["age"].to_numpy(), cfg.AGE_BINS[1:-1])
    out = {"sex:female": pat["sex"].to_numpy() == 0, "sex:male": pat["sex"].to_numpy() == 1}
    for i, lab in enumerate(cfg.AGE_LABELS):
        out[f"age:{lab}"] = age == i
    unit = pat["unit"].to_numpy()
    out["unit:medical"] = unit == 1
    out["unit:surgical"] = unit == 2
    out["unit:not recorded"] = unit == 0
    return out


def analyse(T, S):
    """T = target (evaluation) site, S = source site of the external model."""
    res = {"target": T.name, "source": S.name, "n_patients": T.n_pat, "n_rows": int(len(T.y)),
           "n_septic": int(T.septic.sum()), "row_prevalence": float(T.y.mean())}
    ev = mt.SiteEval(T.y, T.pidx, n_boot=cfg.N_BOOT, seed=cfg.SEED + ord(T.name))
    names = {("local", a): f"int_{T.name}_{a}" for a in cfg.ARMS}
    names.update({("external", a): f"ext_{S.name}to{T.name}_{a}" for a in cfg.ARMS})
    preds, hists, point, boot = {}, {}, {}, {}
    disc = {}
    for key, nm in names.items():
        s = load(nm)
        if s is None:
            continue
        preds[key] = s
        h = hists[key] = ev.hist(s)
        d = ev.discrimination(s, h)
        disc[f"{key[0]}:{key[1]}"] = d
        point[key] = (d["auroc"], d["auprc"])
        boot[key] = (ev._auc(h["POS"], h["NEG"]), ev._ap(h["POS"], h["NEG"]))
    res["discrimination"] = disc

    def contrast(coefs, m=0):
        if any(k not in point for k in coefs):
            return None
        est = sum(w * point[k][m] for k, w in coefs.items())
        b = sum(w * boot[k][m] for k, w in coefs.items())
        return dict(est=float(est), ci=mt.pct_ci(b), p=pval(b))

    L, E = "local", "external"
    res["hypotheses"] = {
        "H1_external_minus_local_VP": contrast({(E, "VP"): 1, (L, "VP"): -1}),
        "H2_P_minus_C_local": contrast({(L, "P"): 1, (L, "C"): -1}),
        "H3_loss_P_minus_loss_V": contrast({(L, "P"): 1, (E, "P"): -1, (L, "V"): -1, (E, "V"): 1}),
    }
    res["transport_loss"] = {a: dict(auroc=contrast({(L, a): 1, (E, a): -1}),
                                     auprc=contrast({(L, a): 1, (E, a): -1}, 1)) for a in cfg.ARMS}
    dec = {}
    for sc in (L, E):
        dec[sc] = dict(
            total_VP_minus_C=contrast({(sc, "VP"): 1, (sc, "C"): -1}),
            values_alone=contrast({(sc, "V"): 1, (sc, "C"): -1}),
            process_alone=contrast({(sc, "P"): 1, (sc, "C"): -1}),
            values_given_process=contrast({(sc, "VP"): 1, (sc, "P"): -1}),
            process_given_values=contrast({(sc, "VP"): 1, (sc, "V"): -1}),
            shapley_values=contrast({(sc, "V"): .5, (sc, "C"): -.5, (sc, "VP"): .5, (sc, "P"): -.5}),
            shapley_process=contrast({(sc, "P"): .5, (sc, "C"): -.5, (sc, "VP"): .5, (sc, "V"): -.5}))
        if all(k in boot for k in [(sc, a) for a in cfg.ARMS]):
            num = .5 * (boot[(sc, "P")][0] - boot[(sc, "C")][0] + boot[(sc, "VP")][0] - boot[(sc, "V")][0])
            den = boot[(sc, "VP")][0] - boot[(sc, "C")][0]
            pt = dec[sc]["shapley_process"]["est"] / dec[sc]["total_VP_minus_C"]["est"]
            dec[sc]["process_share"] = dict(est=float(pt), ci=mt.pct_ci(num / den))
    res["decomposition"] = dec

    # ---- calibration, net benefit, utility, alerting ------------------------------
    pooled = load(f"pool_{T.name}_VP")
    if pooled is not None:
        preds[("pooled", "VP")] = pooled
        hists[("pooled", "VP")] = hp = ev.hist(pooled)
        res["pooled"] = dict(discrimination=ev.discrimination(pooled, hp),
                             delta_vs_local=ev.delta_auc(pooled, None, hp, hists[(L, "VP")]),
                             delta_vs_external=ev.delta_auc(pooled, None, hp, hists[(E, "VP")]))
    res["calibration"] = {f"{k[0]}:{k[1]}": ev.calibration(preds[k], hists[k])
                          for k in preds if k[1] == "VP"}
    res["net_benefit"] = {f"{k[0]}:{k[1]}": ev.net_benefit(preds[k], cfg.NB_THRESHOLDS)
                          for k in preds if k[1] == "VP"}

    u1, u0, best = mt.utility_weights(T.y, T.pidx, T.t)
    su1, su0, sbest = mt.utility_weights(S.y, S.pidx, S.t)
    al_T, al_S = mt.AlertEval(T.y, T.pidx, T.t, W=ev.W), mt.AlertEval(S.y, S.pidx, S.t)
    util, alert = {}, {}
    for a in cfg.ARMS:
        if (L, a) not in preds or (E, a) not in preds:
            continue
        dev = load(f"int_{S.name}_{a}")            # development-site out-of-fold predictions
        loc, ext = preds[(L, a)], preds[(E, a)]
        th_dev, u_dev = mt.best_utility_threshold(dev, su1, su0, sbest)
        th_loc, _ = mt.best_utility_threshold(loc, u1, u0, best)
        th_orc, _ = mt.best_utility_threshold(ext, u1, u0, best)
        util[a] = dict(
            threshold_dev=th_dev, utility_dev_internal=u_dev,
            external_at_dev_threshold=mt.utility_ci(ext, th_dev, u1, u0, best, ev),
            external_at_retuned_threshold=mt.utility_ci(ext, th_orc, u1, u0, best, ev),
            local=mt.utility_ci(loc, th_loc, u1, u0, best, ev))
        tau_dev = al_S.threshold_for_sensitivity(dev, cfg.SENS_TARGET)
        tau_loc = al_T.threshold_for_sensitivity(loc, cfg.SENS_TARGET)
        tau_re = al_T.threshold_for_sensitivity(ext, cfg.SENS_TARGET)
        alert[a] = dict(development_internal=al_S.evaluate(dev, tau_dev),
                        external_at_dev_threshold=al_T.evaluate(ext, tau_dev),
                        external_at_retuned_threshold=al_T.evaluate(ext, tau_re),
                        local=al_T.evaluate(loc, tau_loc))
    res["utility"], res["alerting"] = util, alert

    # ---- subgroups (full model) -----------------------------------------------------
    sub = {}
    if (L, "VP") in preds and (E, "VP") in preds:
        tau_dev = alert["VP"]["external_at_dev_threshold"]["tau"]
        tau_loc = alert["VP"]["local"]["tau"]
        for name, mask in subgroup_masks(T).items():
            if mask.sum() < 50 or T.septic[mask].sum() < 20:
                continue
            se = mt.SiteEval(T.y, T.pidx, n_boot=cfg.N_BOOT, seed=cfg.SEED + 7, patient_mask=mask)
            hl, he = se.hist(preds[(L, "VP")]), se.hist(preds[(E, "VP")])
            als = mt.AlertEval(se.y, se.pidx, T.t[se.rows], W=se.W)
            sub[name] = dict(
                n_patients=int(mask.sum()), n_septic=int(T.septic[mask].sum()),
                local=se.discrimination(None, hl), external=se.discrimination(None, he),
                delta_ext_minus_local=se.delta_auc(None, None, he, hl),
                oe_local=se.calibration(None, hl)["oe"], oe_external=se.calibration(None, he)["oe"],
                alert_local=als.evaluate(se.sub(preds[(L, "VP")]), tau_loc),
                alert_external=als.evaluate(se.sub(preds[(E, "VP")]), tau_dev))
    res["subgroups"] = sub

    # ---- sensitivity: early-onset patients and post-onset hours -----------------------
    sens = {}
    if (L, "VP") in preds:
        keep_pat = T.pat["first_pos"].to_numpy() != 0
        se = mt.SiteEval(T.y, T.pidx, n_boot=cfg.N_BOOT, seed=cfg.SEED + 9, patient_mask=keep_pat)
        hs = {k: se.hist(preds[k]) for k in preds if k[0] in (L, E)}
        sens["excluding_patients_positive_at_first_hour"] = dict(
            n_patients=int(keep_pat.sum()), n_septic=int(T.septic[keep_pat].sum()),
            auroc={f"{k[0]}:{k[1]}": se.discrimination(None, h)["auroc"] for k, h in hs.items()},
            H1=se.delta_auc(None, None, hs[(E, "VP")], hs[(L, "VP")]))
        onset = al_T.onset[T.pidx]
        rows = np.flatnonzero(T.t <= onset)                 # drop hours after onset
        se2 = mt.SiteEval(T.y[rows], T.pidx[rows], n_boot=cfg.N_BOOT, seed=cfg.SEED + ord(T.name))
        hs2 = {k: se2.hist(preds[k][rows]) for k in preds if k[0] in (L, E)}
        sens["excluding_hours_after_onset"] = dict(
            n_rows=int(len(rows)),
            auroc={f"{k[0]}:{k[1]}": se2.discrimination(None, h)["auroc"] for k, h in hs2.items()},
            H1=se2.delta_auc(None, None, hs2[(E, "VP")], hs2[(L, "VP")]))
    res["sensitivity"] = sens

    # ---- exploratory: discrimination by time to onset ---------------------------------
    hor = {}
    septic_row = T.septic[T.pidx] == 1
    d = T.t - al_T.onset[T.pidx].astype(float)
    d[~septic_row] = np.nan
    for lab, lo, hi in (("-24 to -13 h", -24, -13), ("-12 to -7 h", -12, -7),
                        ("-6 to -1 h", -6, -1), ("0 to +3 h", 0, 3)):
        rows = np.flatnonzero(~septic_row | ((d >= lo) & (d <= hi)))
        yy = septic_row[rows].astype(np.int8)
        if yy.sum() < 100:
            continue
        seh = mt.SiteEval(yy, T.pidx[rows], n_boot=cfg.N_BOOT, seed=cfg.SEED + ord(T.name))
        hor[lab] = {"n_positive_rows": int(yy.sum())}
        for k in preds:
            if k[0] in (L, E):
                hor[lab][f"{k[0]}:{k[1]}"] = {kk: vv for kk, vv in seh.discrimination(preds[k][rows]).items()
                                              if kk in ("auroc", "auroc_ci")}
    res["by_time_to_onset"] = hor
    return res


def main():
    A, B = Site("A"), Site("B")
    for T, S in ((B, A), (A, B)):
        res = analyse(T, S)
        (cfg.RES / f"main_{T.name}.json").write_text(json.dumps(jsonable(res), indent=1))
        h = res["hypotheses"]
        print(f"target {T.name}:", {k: (round(v["auroc"], 4)) for k, v in res["discrimination"].items()})
        print("   ", {k: (None if v is None else (round(v["est"], 4), [round(x, 4) for x in v["ci"]], v["p"]))
                      for k, v in h.items()})


if __name__ == "__main__":
    main()
