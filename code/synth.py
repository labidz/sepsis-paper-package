"""Synthetic data with the Challenge 2019 schema, used only to test the code.

Writes one ZIP in the original layout (individual PSV files) and one in the
container layout, so both readers are exercised. No real patient data.
Usage: python3 synth.py <out_dir> [n_patients_per_site]
"""
import hashlib
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402


def make_patient(rng, site):
    L = int(rng.integers(10, 90))
    septic = rng.random() < (0.12 if site == "A" else 0.08)
    r1 = int(rng.integers(0, L)) if septic else None
    if septic:
        L = min(L, r1 + 10)
    t = np.arange(L)
    sick = np.zeros(L)
    if septic:
        sick = np.clip((t - (r1 - 12)) / 12.0, 0, 1.5)
    X = np.full((L, len(cfg.CLIN)), np.nan)
    base = dict(HR=80, O2Sat=97, Temp=36.8, SBP=120, MAP=85, DBP=65, Resp=18, EtCO2=35)
    for j, v in enumerate(cfg.VITALS):
        p_obs = {"Temp": 0.3, "EtCO2": 0.0 if site == "A" else 0.1}.get(v, 0.88)
        val = base[v] + rng.normal(0, 5 if v != "Temp" else 0.4, L)
        if v in ("HR", "Resp"):
            val += 12 * sick
        if v in ("SBP", "MAP"):
            val -= 10 * sick
        X[:, j] = np.where(rng.random(L) < p_obs, np.round(val, 1), np.nan)
    for j, v in enumerate(cfg.LABS, start=len(cfg.VITALS)):
        p = (0.05 if site == "A" else 0.08) * (1 + 2.5 * sick)
        val = np.abs(rng.normal(10, 3, L)) + (3 * sick if v in ("Lactate", "WBC") else 0)
        if v == "FiO2":
            val = np.clip(rng.normal(0.4, 0.1, L), 0.21, 1.0)
        X[:, j] = np.where(rng.random(L) < p, np.round(val, 2), np.nan)
    unit = rng.random()
    u1, u2 = (np.nan, np.nan) if unit < 0.4 else ((1.0, 0.0) if unit < 0.7 else (0.0, 1.0))
    dem = np.column_stack([np.full(L, round(rng.uniform(20, 90), 2)), np.full(L, rng.integers(0, 2)),
                           np.full(L, u1), np.full(L, u2),
                           np.full(L, round(-rng.exponential(30), 2)), t + 1])
    y = np.zeros(L)
    if septic:
        y[r1:] = 1
    return np.column_stack([X, dem, y])


def to_psv(arr):
    lines = ["|".join(cfg.RAW_COLS)]
    for row in arr:
        lines.append("|".join("NaN" if np.isnan(v) else repr(float(v)) for v in row))
    return ("\n".join(lines) + "\n").encode()


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    rng = np.random.default_rng(7)
    with zipfile.ZipFile(out / "synth_setA.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for i in range(1, n + 1):
            z.writestr(f"training/p{i:06d}.psv", to_psv(make_patient(rng, "A")))
    chunks = []
    for i in range(1, n + 1):
        body = to_psv(make_patient(rng, "B"))
        chunks.append(f"#FILE p{100000 + i:06d}.psv {len(body)} {hashlib.md5(body).hexdigest()}\n".encode() + body)
    with zipfile.ZipFile(out / "synth_setB.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("synth_setB.txt", b"".join(chunks))
    print("wrote", out)


if __name__ == "__main__":
    main()
