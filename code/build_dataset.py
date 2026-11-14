"""Read the Challenge 2019 PSV files straight from ZIP archives (no extraction)
and write one tidy parquet file per hospital system.

Two archive layouts are accepted:
  * a ZIP holding the individual pXXXXXX.psv files (the original distribution);
  * a ZIP holding one text container in which every PSV file is preceded by a
    line "#FILE <name> <bytes> <md5>" (files packed from the PhysioNet open-data
    bucket on Amazon S3). The MD5 (the S3 ETag) and byte length of every file
    are verified.

Hospital system is taken from the file name: p0xxxxx = A, p1xxxxx = B.
Usage: python3 -I build_dataset.py archive1.zip [archive2.zip ...]
"""
import hashlib
import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402


def iter_psv(path, stats):
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError(f"corrupt archive {path}")
        names = z.namelist()
        psv = sorted(n for n in names if n.lower().endswith(".psv"))
        if psv:
            for n in psv:
                yield n.rsplit("/", 1)[-1], z.read(n)
            return
        for n in names:
            if not n.lower().endswith(".txt"):
                continue
            buf = z.read(n)
            pos = 0
            while pos < len(buf):
                eol = buf.index(b"\n", pos)
                tag, name, size, md5 = buf[pos:eol].decode().split(" ")
                if tag != "#FILE":
                    raise ValueError(f"bad container record at byte {pos} of {n}")
                size = int(size)
                body = buf[eol + 1: eol + 1 + size]
                if len(body) != size:
                    raise ValueError(f"truncated record {name}")
                if "-" not in md5:
                    if hashlib.md5(body).hexdigest() != md5:
                        raise ValueError(f"MD5 mismatch for {name}")
                    stats["md5_verified"] += 1
                pos = eol + 1 + size
                yield name, body


def main():
    stats = {"md5_verified": 0}
    header_ref = None
    rec = {"A": ([], [], []), "B": ([], [], [])}     # bodies, pids, nrows
    seen = set()
    for path in sys.argv[1:]:
        for name, raw in iter_psv(path, stats):
            if name in seen:
                raise ValueError(f"duplicate file {name}")
            seen.add(name)
            lines = raw.split(b"\n")
            header = lines[0].strip()
            if header_ref is None:
                header_ref = header
            elif header != header_ref:
                raise ValueError(f"header mismatch in {name}")
            body = [ln for ln in lines[1:] if ln.strip()]
            if not body:
                continue
            pid = int(name[:-4].lstrip("pP"))
            bodies, pids, nrows = rec["A" if pid < 100000 else "B"]
            bodies.append(b"\n".join(body))
            pids.append(pid)
            nrows.append(len(body))
    cols = header_ref.decode().split("|")
    if cols != cfg.RAW_COLS:
        raise ValueError(f"unexpected columns: {cols}")
    meta = {"archives": sys.argv[1:], "md5_verified_files": stats["md5_verified"]}
    for site, (bodies, pids, nrows) in rec.items():
        if not bodies:
            continue
        df = pd.read_csv(io.BytesIO(b"\n".join(bodies)), sep="|", header=None,
                         names=cols, dtype=np.float32, na_values=["NaN"])
        nrows = np.asarray(nrows)
        if len(df) != nrows.sum():
            raise ValueError("row count mismatch after parsing")
        df.insert(0, "pid", np.repeat(np.asarray(pids, dtype=np.int32), nrows))
        df.insert(1, "t", np.concatenate([np.arange(k, dtype=np.int32) for k in nrows]))
        df = df.sort_values(["pid", "t"], kind="stable").reset_index(drop=True)
        df.to_parquet(cfg.DATA / f"site_{site}.parquet", index=False)
        y = df["SepsisLabel"].to_numpy()
        pat = df.groupby("pid")["SepsisLabel"].max()
        meta[site] = dict(
            n_patients=int(df["pid"].nunique()), n_rows=int(len(df)),
            n_septic_patients=int(pat.sum()),
            septic_patient_pct=round(float(pat.mean() * 100), 3),
            positive_rows=int(y.sum()), positive_row_pct=round(float(y.mean() * 100), 3),
            pid_min=int(df["pid"].min()), pid_max=int(df["pid"].max()),
            median_hours=float(df.groupby("pid").size().median()))
        print(site, json.dumps(meta[site]))
    print("md5-verified files:", stats["md5_verified"])
    (cfg.DATA / "dataset_meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
