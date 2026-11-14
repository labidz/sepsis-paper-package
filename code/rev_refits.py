"""Hypotheses re-estimated for (i) the harmonised-variable variant (variables
whose patient-level availability differed more than twofold between systems
removed, with dependent derived and laboratory-count features) and (ii) four
refits with new random seeds. Writes results/rev_refits.json."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402
import metrics as mt  # noqa: E402
from analysis_main import jsonable, load  # noqa: E402
from rev_extra import contrasts  # noqa: E402
from train import Site  # noqa: E402

L, E = "local", "external"


def preds_for(T, S, tags):
    p = {}
    for a in cfg.ARMS:
        t = tags.get(a, "")
        p[(L, a)] = load(f"int_{T.name}_{a}{t}")
        p[(E, a)] = load(f"ext_{S.name}to{T.name}_{a}{t}")
    return None if any(v is None for v in p.values()) else p


def main():
    A, B = Site("A"), Site("B")
    out = {"harmonised": {}, "sofa": {}, "seeds": {}}
    for T, S in ((B, A), (A, B)):
        ev = mt.SiteEval(T.y, T.pidx, n_boot=cfg.N_BOOT, seed=cfg.SEED + ord(T.name))
        p = preds_for(T, S, {"V": "_harm", "P": "_harm", "VP": "_harm"})
        if p is not None:
            out["harmonised"][T.name] = contrasts(ev, p)
        p = preds_for(T, S, {"P": "_sofa", "VP": "_sofa"})
        if p is not None:
            out["sofa"][T.name] = contrasts(ev, p)
            print(T.name, "sofa", {k: round(v["est"], 4) for k, v in out["sofa"][T.name].items()}, flush=True)
        seeds = {}
        for s in range(5):
            tag = "" if s == 0 else f"_s{s}"
            p = preds_for(T, S, {a: tag for a in cfg.ARMS})
            if p is not None:
                seeds[str(s)] = contrasts(ev, p)
        out["seeds"][T.name] = seeds
        for k in ("H1", "H2", "H3", "process_share_local", "process_share_external"):
            v = [seeds[s][k]["est"] for s in seeds]
            print(T.name, k, "seeds", len(v), np.round(v, 4).tolist(), flush=True)
        if T.name in out["harmonised"]:
            print(T.name, "harmonised", {k: round(v["est"], 4) for k, v in out["harmonised"][T.name].items()}, flush=True)
    (cfg.RES / "rev_refits.json").write_text(json.dumps(jsonable(out), indent=1))


if __name__ == "__main__":
    main()
