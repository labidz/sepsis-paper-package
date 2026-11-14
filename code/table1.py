"""Cohort description for each hospital system (Table 1) and per-variable
measurement availability (supplement). Writes results/table1.json and
results/variable_availability.csv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402


def q(x, d=1):
    a, b, c = np.nanpercentile(x, [25, 50, 75])
    return dict(median=round(float(b), d), q1=round(float(a), d), q3=round(float(c), d))


def main():
    out, avail = {}, {}
    for site in cfg.SITES:
        df = pd.read_parquet(cfg.DATA / f"site_{site}.parquet")
        pat = pd.read_parquet(cfg.FEAT / f"{site}_patients.parquet")
        g = df.groupby("pid", sort=True)
        M = df[cfg.CLIN].notna()
        n = len(pat)
        sep = pat["septic"] == 1
        onset_h = (pat.loc[sep & (pat["first_pos"] > 0), "first_pos"] + pat.loc[sep & (pat["first_pos"] > 0), "iculos_first"] + 6)
        rows_after_first_pos = (pat.loc[sep, "n_rows"] - pat.loc[sep, "first_pos"])
        iculos = df["ICULOS"].to_numpy()
        step = np.diff(iculos)[np.diff(df["pid"].to_numpy()) == 0]
        lab_any = M[cfg.LABS].any(axis=1)
        out[site] = dict(
            patients=n, patient_hours=int(len(df)), stay_hours=q(pat["n_rows"]),
            age_mean=round(float(pat["age"].mean()), 1), age_sd=round(float(pat["age"].std()), 1),
            age_missing=int(pat["age"].isna().sum()),
            male_pct=round(float((pat["sex"] == 1).mean() * 100), 1),
            unit_medical_pct=round(float((pat["unit"] == 1).mean() * 100), 1),
            unit_surgical_pct=round(float((pat["unit"] == 2).mean() * 100), 1),
            unit_not_recorded_pct=round(float((pat["unit"] == 0).mean() * 100), 1),
            hosp_to_icu_hours=q(-pat["hosp_adm_time"]),
            hosp_adm_time_missing=int(pat["hosp_adm_time"].isna().sum()),
            septic_patients=int(sep.sum()), septic_pct=round(float(sep.mean() * 100), 2),
            positive_hours=int(df["SepsisLabel"].sum()),
            positive_hours_pct=round(float(df["SepsisLabel"].mean() * 100), 2),
            septic_positive_at_first_hour=int((sep & (pat["first_pos"] == 0)).sum()),
            onset_hour_after_icu_admission=q(onset_h),
            rows_from_first_positive=q(rows_after_first_pos),
            rows_from_first_positive_max=int(rows_after_first_pos.max()),
            iculos_first_gt1_pct=round(float((pat["iculos_first"] > 1).mean() * 100), 2),
            iculos_nonunit_steps=int((step != 1).sum()),
            vitals_observed_per_hour=round(float(M[cfg.VITALS].sum(axis=1).mean()), 2),
            labs_observed_per_hour=round(float(M[cfg.LABS].sum(axis=1).mean()), 2),
            hours_with_any_lab_pct=round(float(lab_any.mean() * 100), 1),
            lab_results_per_patient_day=round(float(M[cfg.LABS].to_numpy().sum() / (len(df) / 24)), 1),
        )
        ever = M.groupby(df["pid"].to_numpy()).any()
        avail[site] = pd.DataFrame({f"{site}_pct_hours": (M.mean() * 100).round(2),
                                    f"{site}_pct_patients": (ever.mean() * 100).round(2)})
        del g
    av = pd.concat([avail[s] for s in cfg.SITES], axis=1)
    av.index.name = "variable"
    av.to_csv(cfg.RES / "variable_availability.csv")
    low = av.index[(av["A_pct_patients"] < 1) | (av["B_pct_patients"] < 1)].tolist()
    out["variables_under_1pct_patients_either_site"] = low
    (cfg.RES / "table1.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    print(av.to_string())


if __name__ == "__main__":
    main()
