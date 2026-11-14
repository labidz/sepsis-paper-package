"""Compact text dump of every primary result, used to write the manuscript and
to check each number quoted in the text against the analysis output."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402


def f(x, d=3):
    return f"{x:.{d}f}"


def ci(v, d=3, key="est"):
    return f"{v[key]:.{d}f} ({v['ci'][0]:.{d}f} to {v['ci'][1]:.{d}f})"


def main(sections):
    for T in ("B", "A"):
        r = json.load(open(cfg.RES / f"main_{T}.json"))
        print(f"\n=== TARGET {T} (external model from {r['source']}) n_pat={r['n_patients']} rows={r['n_rows']} "
              f"septic={r['n_septic']} prev={r['row_prevalence']:.4f}")
        if "disc" in sections:
            for k, d in r["discrimination"].items():
                print(f" {k:12s} AUROC {f(d['auroc'])} ({f(d['auroc_ci'][0])}-{f(d['auroc_ci'][1])})  "
                      f"AUPRC {f(d['auprc'])} ({f(d['auprc_ci'][0])}-{f(d['auprc_ci'][1])})")
            for k, v in r["hypotheses"].items():
                print(f" {k}: {ci(v, 4)} p={v['p']}")
            for a, v in r["transport_loss"].items():
                print(f" loss[{a}] AUROC {ci(v['auroc'], 4)} | AUPRC {ci(v['auprc'], 4)}")
            for sc, d in r["decomposition"].items():
                print(f" decomp[{sc}] " + " | ".join(f"{k}={ci(v, 4) if 'p' in v or 'ci' in v else v}" for k, v in d.items()))
            p = r["pooled"]
            print(f" pooled VP AUROC {f(p['discrimination']['auroc'])} ({f(p['discrimination']['auroc_ci'][0])}-"
                  f"{f(p['discrimination']['auroc_ci'][1])}) AUPRC {f(p['discrimination']['auprc'])}; "
                  f"vs local {ci(p['delta_vs_local'], 4, 'delta')}; vs external {ci(p['delta_vs_external'], 4, 'delta')}")
        if "cal" in sections:
            for k, c in r["calibration"].items():
                print(f" cal[{k}] O:E {f(c['oe'])} ({f(c['oe_ci'][0])}-{f(c['oe_ci'][1])}) citl {f(c['citl'])} "
                      f"({f(c['citl_ci'][0])}-{f(c['citl_ci'][1])}) slope {f(c['slope'])} ({f(c['slope_ci'][0])}-"
                      f"{f(c['slope_ci'][1])}) ICI {f(c['ici'], 4)} ({f(c['ici_ci'][0], 4)}-{f(c['ici_ci'][1], 4)}) "
                      f"Brier {f(c['brier'], 5)} scaled {f(c['brier_scaled'], 4)}")
            for k, nb in r["net_benefit"].items():
                print(f" NB[{k}] " + " ".join(f"{x['pt']}:{x['nb'] * 1000:.2f}/{x['nb_all'] * 1000:.2f}"
                                                 f"[{x['nb_minus_all_ci'][0] * 1000:.2f},{x['nb_minus_all_ci'][1] * 1000:.2f}]"
                                                 for x in nb) + "  (per 1000 patient-hours: model/treat-all[CI diff])")
        if "alert" in sections:
            for a, u in r["utility"].items():
                print(f" util[{a}] dev_th {f(u['threshold_dev'], 4)} dev_int {f(u['utility_dev_internal'])} | ext@dev "
                      f"{f(u['external_at_dev_threshold'][0])} ({f(u['external_at_dev_threshold'][1][0])},"
                      f"{f(u['external_at_dev_threshold'][1][1])}) | ext@retuned {f(u['external_at_retuned_threshold'][0])} "
                      f"| local {f(u['local'][0])} ({f(u['local'][1][0])},{f(u['local'][1][1])})")
            for a, al in r["alerting"].items():
                for k, e in al.items():
                    extra = (f" sensCI {e['sensitivity_ci'][0]:.3f}-{e['sensitivity_ci'][1]:.3f} faCI "
                             f"{e['false_alert_prop_ci'][0]:.3f}-{e['false_alert_prop_ci'][1]:.3f} ppvCI "
                             f"{e['ppv_ci'][0]:.3f}-{e['ppv_ci'][1]:.3f}") if "sensitivity_ci" in e else ""
                    print(f" alert[{a}][{k}] tau {f(e['tau'], 4)} sens {f(e['sensitivity'])} FA {f(e['false_alert_prop'])} "
                          f"PPV {f(e['ppv'])} lead {e['lead_median']} {e['lead_iqr']} ge6 {f(e['lead_ge6_prop'])}{extra}")
        if "sub" in sections:
            for k, s in r["subgroups"].items():
                print(f" sub[{k}] n={s['n_patients']} sep={s['n_septic']} local {f(s['local']['auroc'])} "
                      f"({f(s['local']['auroc_ci'][0])}-{f(s['local']['auroc_ci'][1])}) ext {f(s['external']['auroc'])} "
                      f"({f(s['external']['auroc_ci'][0])}-{f(s['external']['auroc_ci'][1])}) d {ci(s['delta_ext_minus_local'], 3, 'delta')} "
                      f"O:E loc {f(s['oe_local'], 2)} ext {f(s['oe_external'], 2)} | alert loc sens {f(s['alert_local']['sensitivity'], 2)} "
                      f"FA {f(s['alert_local']['false_alert_prop'], 2)} | ext sens {f(s['alert_external']['sensitivity'], 2)} "
                      f"FA {f(s['alert_external']['false_alert_prop'], 2)}")
        if "sens" in sections:
            for k, s in r["sensitivity"].items():
                print(f" sens[{k}] " + " ".join(f"{a}={v:.3f}" for a, v in s["auroc"].items()) + f" | H1 {ci(s['H1'], 4, 'delta')}")
            for lab, h in r["by_time_to_onset"].items():
                print(f" horizon[{lab}] npos={h['n_positive_rows']} " + " ".join(
                    f"{k}={v['auroc']:.3f}" for k, v in h.items() if k != "n_positive_rows"))


if __name__ == "__main__":
    main(sys.argv[1:] or ["disc", "cal", "alert", "sub", "sens"])
