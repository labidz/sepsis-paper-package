"""Manuscript tables as markdown, generated directly from the analysis output so
that no number is typed by hand. Writes manuscript/tables/*.md and prints the
key figures quoted in the text."""
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "manuscript" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
ARM = {"C": "C: covariates only", "V": "V: values (process concealed)",
       "P": "P: process (no values)", "VP": "VP: values + process"}


def R(T):
    return json.load(open(cfg.RES / f"main_{T}.json"))


def a3(d, k="auroc"):
    return f"{d[k]:.3f} ({d[k + '_ci'][0]:.3f} to {d[k + '_ci'][1]:.3f})"


def c3(v, d=3, key="est"):
    return f"{v[key]:.{d}f} ({v['ci'][0]:.{d}f} to {v['ci'][1]:.{d}f})"


def pct(x, d=1):
    return f"{x * 100:.{d}f}"


def table1():
    t = json.load(open(cfg.RES / "table1.json"))
    A, B = t["A"], t["B"]
    rows = [
        ("Patients, n", f"{A['patients']:,}", f"{B['patients']:,}"),
        ("Patient-hours, n", f"{A['patient_hours']:,}", f"{B['patient_hours']:,}"),
        ("Recorded hours per patient, median (IQR)", *[f"{x['stay_hours']['median']:.0f} ({x['stay_hours']['q1']:.0f} to {x['stay_hours']['q3']:.0f})" for x in (A, B)]),
        ("Age, years, mean (SD)", *[f"{x['age_mean']} ({x['age_sd']})" for x in (A, B)]),
        ("Male sex, %", f"{A['male_pct']}", f"{B['male_pct']}"),
        ("Unit type: medical / surgical / not recorded, %", *[f"{x['unit_medical_pct']} / {x['unit_surgical_pct']} / {x['unit_not_recorded_pct']}" for x in (A, B)]),
        ("Hours from hospital to unit admission, median (IQR)", *[f"{x['hosp_to_icu_hours']['median']} ({x['hosp_to_icu_hours']['q1']} to {x['hosp_to_icu_hours']['q3']})" for x in (A, B)]),
        ("Patients with sepsis, n (%)", *[f"{x['septic_patients']:,} ({x['septic_pct']:.1f})" for x in (A, B)]),
        ("Positive patient-hours, n (%)", *[f"{x['positive_hours']:,} ({x['positive_hours_pct']:.2f})" for x in (A, B)]),
        ("Sepsis label positive at first recorded hour, n", f"{A['septic_positive_at_first_hour']}", f"{B['septic_positive_at_first_hour']}"),
        ("Onset, hours after unit admission, median (IQR)", *[f"{x['onset_hour_after_icu_admission']['median']:.0f} ({x['onset_hour_after_icu_admission']['q1']:.0f} to {x['onset_hour_after_icu_admission']['q3']:.0f})" for x in (A, B)]),
        ("Vital signs recorded per hour, mean (of 8)", f"{A['vitals_observed_per_hour']}", f"{B['vitals_observed_per_hour']}"),
        ("Hours with at least one laboratory result, %", f"{A['hours_with_any_lab_pct']}", f"{B['hours_with_any_lab_pct']}"),
        ("Laboratory results per patient-day, mean", f"{A['lab_results_per_patient_day']}", f"{B['lab_results_per_patient_day']}"),
    ]
    md = "| Characteristic | System A | System B |\n| --- | --- | --- |\n" + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)
    (OUT / "table1.md").write_text(md)


def table2():
    lines = ["| Target site and feature arm | Local model: AUROC | External model: AUROC | Transport loss in AUROC | Local model: AUPRC | External model: AUPRC |",
             "| --- | --- | --- | --- | --- | --- |"]
    for T in ("B", "A"):
        r = R(T)
        lines.append(f"| **Evaluated in system {T}** (external model developed in {r['source']}) | | | | | |")
        for arm in cfg.ARMS:
            lo, ex = r["discrimination"][f"local:{arm}"], r["discrimination"][f"external:{arm}"]
            lines.append(f"| {ARM[arm]} | {a3(lo)} | {a3(ex)} | {c3(r['transport_loss'][arm]['auroc'])} | {a3(lo, 'auprc')} | {a3(ex, 'auprc')} |")
    (OUT / "table2.md").write_text("\n".join(lines))


def table3():
    lines = ["| Measure | System B: local model | System B: external model (from A) | System A: local model | System A: external model (from B) |",
             "| --- | --- | --- | --- | --- |"]
    rb, ra = R("B"), R("A")

    def cal(r, k, m, d=2):
        c = r["calibration"][k]
        return f"{c[m]:.{d}f} ({c[m + '_ci'][0]:.{d}f} to {c[m + '_ci'][1]:.{d}f})"

    def row(label, fn):
        lines.append(f"| {label} | {fn(rb, 'local')} | {fn(rb, 'external')} | {fn(ra, 'local')} | {fn(ra, 'external')} |")

    row("Observed-to-expected ratio", lambda r, s: cal(r, f"{s}:VP", "oe"))
    row("Calibration intercept", lambda r, s: cal(r, f"{s}:VP", "citl"))
    row("Calibration slope", lambda r, s: cal(r, f"{s}:VP", "slope"))
    row("Integrated calibration index, percentage points",
        lambda r, s: "{:.2f} ({:.2f} to {:.2f})".format(*[x * 100 for x in (r['calibration'][f'{s}:VP']['ici'], *r['calibration'][f'{s}:VP']['ici_ci'])]))
    row("Brier score × 100", lambda r, s: "{:.2f} ({:.2f} to {:.2f})".format(*[x * 100 for x in (r['calibration'][f'{s}:VP']['brier'], *r['calibration'][f'{s}:VP']['brier_ci'])]))

    def nb(r, s, pt):
        x = [z for z in r["net_benefit"][f"{s}:VP"] if abs(z["pt"] - pt) < 1e-9][0]
        return f"{x['nb'] * 1000:.1f} ({x['nb_ci'][0] * 1000:.1f} to {x['nb_ci'][1] * 1000:.1f})"
    for pt in (0.02, 0.05):
        row(f"Net benefit at a {pt * 100:.0f}% threshold, per 1,000 patient-hours", lambda r, s, pt=pt: nb(r, s, pt))

    rx = json.load(open(cfg.RES / "rev_extra.json")) if (cfg.RES / "rev_extra.json").exists() else None

    def util(r, s):
        if s == "local" and rx is not None:          # threshold chosen on the other folds
            u = rx[r["target"]]["utility_local_cv_threshold"]
            return f"{u['est']:.3f} ({u['ci'][0]:.3f} to {u['ci'][1]:.3f})"
        u = r["utility"]["VP"]["local" if s == "local" else "external_at_dev_threshold"]
        return f"{u[0]:.3f} ({u[1][0]:.3f} to {u[1][1]:.3f})"
    row("Challenge utility score", util)

    def al(r, s, m):
        e = r["alerting"]["VP"]["local" if s == "local" else "external_at_dev_threshold"]
        return f"{e[m] * 100:.1f} ({e[m + '_ci'][0] * 100:.1f} to {e[m + '_ci'][1] * 100:.1f})"
    row("Septic patients alerted at or before onset, %", lambda r, s: al(r, s, "sensitivity"))
    row("Non-septic patients alerted, %", lambda r, s: al(r, s, "false_alert_prop"))
    row("Positive predictive value of an alert, %", lambda r, s: al(r, s, "ppv"))

    def retuned(r, s):
        if s == "local":
            return "same as above"
        e = r["alerting"]["VP"]["external_at_retuned_threshold"]
        return f"{e['false_alert_prop'] * 100:.1f} ({e['false_alert_prop_ci'][0] * 100:.1f} to {e['false_alert_prop_ci'][1] * 100:.1f})"
    row("Non-septic patients alerted when the threshold is re-set to 80% sensitivity at the target site, %", retuned)
    (OUT / "table3.md").write_text("\n".join(lines))


def facts():
    for T in ("B", "A"):
        r = R(T)
        h, d = r["hypotheses"], r["decomposition"]
        print(f"[{T}] H1 {c3(h['H1_external_minus_local_VP'])} | H2 {c3(h['H2_P_minus_C_local'])} | H3 {c3(h['H3_loss_P_minus_loss_V'])}")
        print(f"[{T}] process share local {d['local']['process_share']['est'] * 100:.0f}% ({d['local']['process_share']['ci'][0] * 100:.0f}-{d['local']['process_share']['ci'][1] * 100:.0f}) "
              f"external {d['external']['process_share']['est'] * 100:.0f}% ({d['external']['process_share']['ci'][0] * 100:.0f}-{d['external']['process_share']['ci'][1] * 100:.0f}); "
              f"proc|values local {c3(d['local']['process_given_values'])} ext {c3(d['external']['process_given_values'])}; "
              f"values|proc local {c3(d['local']['values_given_process'])} ext {c3(d['external']['values_given_process'])}")
        p = r["pooled"]
        print(f"[{T}] pooled {a3(p['discrimination'])} vs local {c3(p['delta_vs_local'], 3, 'delta')}")


if __name__ == "__main__":
    table1(); table2(); table3(); facts()
    for f in sorted(OUT.glob("table[123].md")):
        print(f"\n--- {f.name}\n{f.read_text()}")
