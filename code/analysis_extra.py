"""Alerting at several operating points (supplementary Table C.9).

For each sensitivity target fixed at the development site, report what the
transported threshold does at the target site, and the proportion of
non-septic patients alerted by the local and external models when both are
set to that sensitivity at the target site. Full model (VP) only.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402
import metrics as mt  # noqa: E402
from train import Site  # noqa: E402

TARGETS = [0.5, 0.6, 0.7, 0.8, 0.9]


def main():
    A, B = Site("A"), Site("B")
    out, lines = {}, ["| Direction | Sensitivity target at development site, % | External site, development threshold: septic alerted, % | External site, development threshold: non-septic alerted, % | Non-septic alerted at the target sensitivity, external model, % | Non-septic alerted at the target sensitivity, local model, % |",
                      "| --- | --- | --- | --- | --- | --- |"]
    for T, S in ((B, A), (A, B)):
        dev = np.load(cfg.PRED / f"int_{S.name}_VP.npy").astype(float)
        loc = np.load(cfg.PRED / f"int_{T.name}_VP.npy").astype(float)
        ext = np.load(cfg.PRED / f"ext_{S.name}to{T.name}_VP.npy").astype(float)
        al_S, al_T = mt.AlertEval(S.y, S.pidx, S.t), mt.AlertEval(T.y, T.pidx, T.t)
        key = f"{S.name}to{T.name}"
        out[key] = {}
        for tg in TARGETS:
            e_dev = al_T.evaluate(ext, al_S.threshold_for_sensitivity(dev, tg))
            e_ext = al_T.evaluate(ext, al_T.threshold_for_sensitivity(ext, tg))
            e_loc = al_T.evaluate(loc, al_T.threshold_for_sensitivity(loc, tg))
            out[key][str(tg)] = dict(ext_dev_sens=e_dev["sensitivity"], ext_dev_fa=e_dev["false_alert_prop"],
                                     ext_retuned_fa=e_ext["false_alert_prop"], local_fa=e_loc["false_alert_prop"],
                                     ext_retuned_ppv=e_ext["ppv"], local_ppv=e_loc["ppv"])
            lines.append(f"| {S.name} to {T.name} | {tg * 100:.0f} | {e_dev['sensitivity'] * 100:.1f} | {e_dev['false_alert_prop'] * 100:.1f} | "
                         f"{e_ext['false_alert_prop'] * 100:.1f} | {e_loc['false_alert_prop'] * 100:.1f} |")
    (cfg.RES / "alert_operating_points.json").write_text(json.dumps(out, indent=1))
    (Path(__file__).resolve().parents[1] / "manuscript" / "tables" / "tableC9.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
