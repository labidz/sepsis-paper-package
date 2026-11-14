"""Supplementary tables (markdown) generated from the analysis output."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "manuscript" / "tables"


def R(T):
    return json.load(open(cfg.RES / f"main_{T}.json"))


def e3(v, key="est", d=3):
    return "not available" if v is None else f"{v[key]:.{d}f} ({v['ci'][0]:.{d}f} to {v['ci'][1]:.{d}f})"


def loss(v, key="est", d=3):
    """Transport loss (local minus external) from an external-minus-local contrast."""
    return "not available" if v is None else f"{-v[key]:.{d}f} ({-v['ci'][1]:.{d}f} to {-v['ci'][0]:.{d}f})"


def hu(x):
    """Round half up (avoids half-to-even rounding of medians such as 5.5 and 90.5)."""
    return f"{int(np.floor(x + 0.5)):,}"


def c1():
    lines = ["| Measure | System B | System A |", "| --- | --- | --- |"]
    rb, ra = R("B"), R("A")

    def row(lab, fn):
        lines.append(f"| {lab} | {fn(rb)} | {fn(ra)} |")
    row("AUROC, two-site model", lambda r: f"{r['pooled']['discrimination']['auroc']:.3f} ({r['pooled']['discrimination']['auroc_ci'][0]:.3f} to {r['pooled']['discrimination']['auroc_ci'][1]:.3f})")
    row("AUROC, difference from local model", lambda r: e3(r["pooled"]["delta_vs_local"], "delta"))
    row("AUROC, difference from external model", lambda r: e3(r["pooled"]["delta_vs_external"], "delta"))
    row("AUPRC, two-site model", lambda r: f"{r['pooled']['discrimination']['auprc']:.3f} ({r['pooled']['discrimination']['auprc_ci'][0]:.3f} to {r['pooled']['discrimination']['auprc_ci'][1]:.3f})")
    row("Observed-to-expected ratio", lambda r: f"{r['calibration']['pooled:VP']['oe']:.2f} ({r['calibration']['pooled:VP']['oe_ci'][0]:.2f} to {r['calibration']['pooled:VP']['oe_ci'][1]:.2f})")
    row("Calibration slope", lambda r: f"{r['calibration']['pooled:VP']['slope']:.2f} ({r['calibration']['pooled:VP']['slope_ci'][0]:.2f} to {r['calibration']['pooled:VP']['slope_ci'][1]:.2f})")
    (OUT / "tableC1.md").write_text("\n".join(lines))


def c2():
    s = json.load(open(cfg.RES / "sensitivity.json"))
    names = {"primary": "Primary analysis", "values_median_imputation": "Values arm: median imputation instead of random draws",
             "values_current_only": "Values arm: current values only (no trailing statistics)",
             "excluding_EtCO2": "End-tidal CO2 excluded from all arms",
             "logistic_regression": "Logistic regression as the learner (all arms)",
             "tuned_hyperparameters": "Full model with hyperparameter search"}
    lines = ["| Specification | Target | H1: transport loss of the full model (local minus external) | H2: process minus covariates, local | H3: excess loss of process over values | Process share of gain, local / external |",
             "| --- | --- | --- | --- | --- | --- |"]
    for k, lab in names.items():
        for T in ("B", "A"):
            if k not in s or T not in s[k]:
                continue
            r = s[k][T]
            sh = "not available"
            if "process_share_local" in r and "process_share_external" in r:
                sh = f"{r['process_share_local']['est'] * 100:.0f}% / {r['process_share_external']['est'] * 100:.0f}%"
            lines.append(f"| {lab} | {T} | {loss(r['H1'])} | {e3(r['H2'])} | {e3(r['H3'])} | {sh} |")
    sub = json.load(open(cfg.RES / "sens_subsets.json")) if (cfg.RES / "sens_subsets.json").exists() else {}
    for k, lab in (("excluding_patients_positive_at_first_hour", "Patients positive at first recorded hour excluded"),
                   ("excluding_hours_after_onset", "Hours after onset excluded")):
        for T in ("B", "A"):
            if T in sub:
                r = sub[T][k]
                sh = f"{r['process_share_local']['est'] * 100:.0f}% / {r['process_share_external']['est'] * 100:.0f}%"
                lines.append(f"| {lab} | {T} | {loss(r['H1'])} | {e3(r['H2'])} | {e3(r['H3'])} | {sh} |")
            else:
                r = R(T)["sensitivity"]
                lines.append(f"| {lab} | {T} | {loss(r[k]['H1'], 'delta')} | not re-estimated | not re-estimated | not re-estimated |")
    rr = json.load(open(cfg.RES / "rev_refits.json")) if (cfg.RES / "rev_refits.json").exists() else {}
    for T in ("B", "A"):
        h = rr.get("harmonised", {}).get(T)
        if h:
            sh = f"{h['process_share_local']['est'] * 100:.0f}% / {h['process_share_external']['est'] * 100:.0f}%"
            lines.append(f"| Variables with more than twofold difference in availability removed (10 variables) | {T} | {loss(h['H1'])} | {e3(h['H2'])} | {e3(h['H3'])} | {sh} |")
    for T in ("B", "A"):
        h = rr.get("sofa", {}).get(T)
        if h:
            sh = f"{h['process_share_local']['est'] * 100:.0f}% / {h['process_share_external']['est'] * 100:.0f}%"
            lines.append(f"| Process features of organ-dysfunction variables, all measured-this-hour flags and laboratory counts removed | {T} | {loss(h['H1'])} | {e3(h['H2'])} | {e3(h['H3'])} | {sh} |")
    for T in ("B", "A"):
        sd = rr.get("seeds", {}).get(T, {})
        if len(sd) > 1:
            def mr(k, sign=1, pct=False):
                v = np.array([sd[s][k]["est"] for s in sd]) * sign * (100 if pct else 1)
                f = "{:.0f}%" if pct else "{:.3f}"
                return f"{f.format(np.median(v))} ({f.format(v.min())} to {f.format(v.max())})"
            lines.append(f"| Refitted with new random seeds: median (range) over {len(sd)} fits | {T} | {mr('H1', -1)} | {mr('H2')} | {mr('H3')} | {mr('process_share_local', pct=True)} / {mr('process_share_external', pct=True)} |")
    g = s.get("recurrent_network_fold0", {})
    for T in ("B", "A"):
        if T in g:
            lines.append(f"| Recurrent network (one test fold, {g[T]['n_patients']:,} patients) | {T} | {loss(g[T]['H1_gru'], 'delta')}; boosted model on the same patients {loss(g[T]['H1_gbm'], 'delta')} | not applicable | not applicable | not applicable |")
    (OUT / "tableC2.md").write_text("\n".join(lines))
    return s


def c3():
    lines = ["| Target | Subgroup | Patients (septic), n | Local AUROC | External AUROC | External minus local | O:E local / external | Septic alerted, local / external, % | Non-septic alerted, local / external, % |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for T in ("B", "A"):
        for k, s in R(T)["subgroups"].items():
            lab = k.replace("sex:", "Sex: ").replace("age:", "Age: ").replace("unit:", "Unit: ").replace(">=80", "80 or older").replace("<50", "under 50")
            lines.append(f"| {T} | {lab} | {s['n_patients']:,} ({s['n_septic']:,}) | {s['local']['auroc']:.3f} ({s['local']['auroc_ci'][0]:.3f} to {s['local']['auroc_ci'][1]:.3f}) | "
                         f"{s['external']['auroc']:.3f} ({s['external']['auroc_ci'][0]:.3f} to {s['external']['auroc_ci'][1]:.3f}) | {e3(s['delta_ext_minus_local'], 'delta')} | "
                         f"{s['oe_local']:.2f} / {s['oe_external']:.2f} | {s['alert_local']['sensitivity'] * 100:.0f} / {s['alert_external']['sensitivity'] * 100:.0f} | "
                         f"{s['alert_local']['false_alert_prop'] * 100:.0f} / {s['alert_external']['false_alert_prop'] * 100:.0f} |")
    (OUT / "tableC3.md").write_text("\n".join(lines))


def c4():
    av = pd.read_csv(cfg.RES / "variable_availability.csv", index_col=0)
    lines = ["| Variable | A: patients with a measurement, % | B: patients with a measurement, % | A: hours with a measurement, % | B: hours with a measurement, % |", "| --- | --- | --- | --- | --- |"]
    for v, r in av.iterrows():
        lines.append(f"| {v} | {r['A_pct_patients']:.1f} | {r['B_pct_patients']:.1f} | {r['A_pct_hours']:.1f} | {r['B_pct_hours']:.1f} |")
    (OUT / "tableC4.md").write_text("\n".join(lines))


def c5():
    u = json.load(open(cfg.RES / "updating.json"))
    lines = ["| Direction | Local patients used | Septic patients in sample, median | ICI before, percentage points | ICI after, median (2.5th to 97.5th percentile) | Repetitions improved, % | O:E after | Slope after | Septic alerted with locally re-set threshold, % | Non-septic alerted with locally re-set threshold, % |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, lab in (("AtoB", "A to B"), ("BtoA", "B to A")):
        for n, d in u[k].items():
            a, b, al = d["after"], d["before"], d["alerting"]
            rng = lambda x, m=1, dd=2: f"{x['median'] * m:.{dd}f} ({x['lo'] * m:.{dd}f} to {x['hi'] * m:.{dd}f})"
            lines.append(f"| {lab} | {int(n):,} | {hu(a['n_septic_in_sample']['median'])} | {b['ici']['median'] * 100:.2f} | {rng(a['ici'], 100)} | {d['ici_improved_prop'] * 100:.0f} | "
                         f"{rng(a['oe'])} | {rng(a['slope'])} | {rng(al['sens_local'], 100, 0) if al else 'not estimated'} | {rng(al['fa_local'], 100, 0) if al else 'not estimated'} |")
    (OUT / "tableC5.md").write_text("\n".join(lines))


def c6():
    sh = json.load(open(cfg.RES / "shap.json"))
    lines = ["| Rank | Model developed in A: feature (family), share % | Model developed in B: feature (family), share % |", "| --- | --- | --- |"]
    ta, tb = sh["model_A_on_A"]["top20"], sh["model_B_on_B"]["top20"]
    for i in range(12):
        f = lambda t: f"{t['feature'].replace('_', ' ')} ({t['family']}), {t['share'] * 100:.1f}"
        lines.append(f"| {i + 1} | {f(ta[i])} | {f(tb[i])} |")
    fs = lambda k: " / ".join(f"{sh[k]['family_share'][x] * 100:.0f}" for x in ("process", "value", "covariate"))
    lines.append(f"| Family shares on development data (process / value / covariate), % | {fs('model_A_on_A')} | {fs('model_B_on_B')} |")
    lines.append(f"| Family shares on the other system's data, % | {fs('model_A_on_B')} | {fs('model_B_on_A')} |")
    (OUT / "tableC6.md").write_text("\n".join(lines))
    return sh


def c7():
    lines = ["| Target | Hours relative to onset | Positive hours, n | C local / external | V local / external | P local / external | VP local / external |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for T in ("B", "A"):
        for lab, h in R(T)["by_time_to_onset"].items():
            cell = lambda a: f"{h['local:' + a]['auroc']:.3f} / {h['external:' + a]['auroc']:.3f}"
            lines.append(f"| {T} | {lab} | {h['n_positive_rows']:,} | {cell('C')} | {cell('V')} | {cell('P')} | {cell('VP')} |")
    (OUT / "tableC7.md").write_text("\n".join(lines))


def fits():
    rows = [json.loads(l) for l in open(cfg.RES / "train_log.jsonl")]
    df = pd.DataFrame(rows)
    df = df[df["tag"] == ""]
    g = df.groupby(["stage", "arm"]).agg(fits=("best_iter", "size"), trees_median=("best_iter", "median"),
                                         trees_min=("best_iter", "min"), trees_max=("best_iter", "max"),
                                         features=("n_features", "first"), train_rows_median=("n_train_rows", "median"))
    pats = {s: pd.read_parquet(cfg.FEAT / f"{s}_patients.parquet") for s in "AB"}
    ev_ext = {s: int(pats[s]["septic"].sum()) for s in "AB"}
    ev_int = {s: int(np.median([pats[s]["septic"][pats[s]["fold"] != k].sum() for k in range(cfg.N_FOLDS)])) for s in "AB"}
    ev_pool = int(np.median([sum(pats[s]["septic"][pats[s]["fold"] != k].sum() for s in "AB") for k in range(cfg.N_FOLDS)]))
    lines = ["| Stage | Arm | Fits, n | Features, n | Trees at early stopping, median (range) | Training patient-hours, median | Septic patients in the development data, n |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for (st, arm), r in g.iterrows():
        evs = {"external": f"{ev_ext['A']:,} (A) / {ev_ext['B']:,} (B)", "internal": f"{ev_int['A']:,} (A) / {ev_int['B']:,} (B), median per fold", "pooled": f"{ev_pool:,}, median per fold"}[st]
        lines.append(f"| {st} | {arm} | {int(r['fits'])} | {int(r['features'])} | {r['trees_median']:.0f} ({int(r['trees_min'])} to {int(r['trees_max'])}) | {r['train_rows_median']:,.0f} | {evs} |")
    (OUT / "tableC8.md").write_text("\n".join(lines))


def c10():
    rx = json.load(open(cfg.RES / "rev_extra.json"))
    lines = ["| Target | Hours relative to onset | H1: transport loss, full model | H2: process minus covariates, local | H3: excess loss of process over values | Process share of gain, local / external |",
             "| --- | --- | --- | --- | --- | --- |"]
    for T in ("B", "A"):
        for w, r in rx[T]["by_window"].items():
            lines.append(f"| {T} | {w} | {loss(r['H1'])} | {e3(r['H2'])} | {e3(r['H3'])} | "
                         f"{r['process_share_local']['est'] * 100:.0f}% / {r['process_share_external']['est'] * 100:.0f}% |")
    (OUT / "tableC10.md").write_text("\n".join(lines))


def c11():
    f = cfg.RES / "localrefit.json"
    if not f.exists():
        return
    lr = json.load(open(f))
    lines = ["| Target | Local patients used (septic, median) | New local model, AUROC median (range) | External model updated by further boosting, AUROC median (range) | Trees added, median |",
             "| --- | --- | --- | --- | --- |"]
    for T in ("B", "A"):
        r = lr[T]
        lines.append(f"| {T} | Reference: external model unchanged | {r['auroc_external']:.3f} | | |")
        lines.append(f"| {T} | Reference: cross-validated local model (about 13,000 patients) | {r['auroc_local_cv']:.3f} | | |")
        for n, rec in r["sizes"].items():
            a = np.array([x["auroc_new"] for x in rec]); c = np.array([x["auroc_continued"] for x in rec])
            ns = np.median([x["n_septic"] for x in rec]); tr = np.median([x["trees_added"] for x in rec])
            lines.append(f"| {T} | {int(n):,} ({hu(ns)}) | {np.median(a):.3f} ({a.min():.3f} to {a.max():.3f}) | "
                         f"{np.median(c):.3f} ({c.min():.3f} to {c.max():.3f}) | {hu(tr)} |")
    (OUT / "tableC11.md").write_text("\n".join(lines))


if __name__ == "__main__":
    c1(); s = c2(); c3(); c4(); c5(); c6(); c7(); fits(); c10(); c11()
    for n in sys.argv[1:]:
        print(f"--- tableC{n}\n" + (OUT / f"tableC{n}.md").read_text())
