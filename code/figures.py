"""Publication figures (PNG at 600 dpi and vector PDF).

Colours are a validated colour-blind-safe palette; identity is never carried by
colour alone (marker shape, line style or hatching repeat it) so the figures
also read in greyscale.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.facecolor": "white", "figure.facecolor": "white"})
ARM_LABEL = {"C": "Covariates\nonly", "V": "Values", "P": "Process", "VP": "Values +\nprocess"}
SITE_NAME = {"A": "A", "B": "B"}


def load(T):
    return json.load(open(cfg.RES / f"main_{T}.json"))


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(cfg.FIG / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, s):
    ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom")


# ------------------------------------------------------------------ Figure 1
def fig1():
    fig, ax = plt.subplots(figsize=(7.1, 3.1))
    ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")

    def box(x, y, w, h, text, fc="white", ec=MUTED, fs=7.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8",
                                    fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=INK, linespacing=1.35)

    ax.text(0, 42.5, "a  Feature arms (2 × 2 factorial)", fontsize=8.5, fontweight="bold")
    ax.text(27.2, 38.4, "Measurement process", ha="center", fontsize=7.2, color=MUTED)
    ax.text(17, 35.6, "not used", ha="center", fontsize=7.2, color=MUTED)
    ax.text(37.5, 35.6, "used", ha="center", fontsize=7.2, color=MUTED)
    ax.text(1.0, 22, "Physiological values", rotation=90, va="center", fontsize=7.2, color=MUTED)
    ax.text(4.6, 28.5, "not used", rotation=90, va="center", fontsize=7.2, color=MUTED)
    ax.text(4.6, 15.5, "used", rotation=90, va="center", fontsize=7.2, color=MUTED)
    box(8, 23, 18, 11, "C\ncovariates only", fs=6.8)
    box(28.5, 23, 18, 11, "P\nmeasurement timing\nand frequency\n(139 features)", fs=6.3)
    box(8, 10, 18, 11, "V\nmeasured values,\nstatus concealed\n(75 features)", fs=6.8)
    box(28.5, 10, 18, 11, "VP\nvalues + process\n(full model)", fs=6.8)
    ax.text(27.2, 4.4, "All arms include five covariates: age, sex, unit type,\nhours from hospital to unit admission, hours since unit admission",
            ha="center", fontsize=6.6, color=MUTED, linespacing=1.3)

    ax.text(53, 42.5, "b  Validation (shown for target B; mirrored for A)", fontsize=8.5, fontweight="bold")
    box(53, 27, 18, 9, "Hospital system A\n20,336 patients", fc="#fde9e0", ec=ORANGE, fs=6.8)
    box(53, 9, 18, 9, "Hospital system B\n20,000 patients", fc="#e3effc", ec=BLUE, fs=6.8)
    box(79, 27, 20, 9, "External model\ndeveloped on all of A", fc="#fde9e0", ec=ORANGE, fs=6.8)
    box(79, 9, 20, 9, "Local model\n5-fold cross-validation\nwithin B", fc="#e3effc", ec=BLUE, fs=6.8)
    kw = dict(arrowstyle="-|>", lw=0.9, color=MUTED, shrinkA=3, shrinkB=3)
    ax.annotate("", xy=(79, 31.5), xytext=(71, 31.5), arrowprops=kw)
    ax.annotate("", xy=(79, 13.5), xytext=(71, 13.5), arrowprops=kw)
    ax.annotate("", xy=(89, 24.2), xytext=(89, 27), arrowprops=dict(kw, color=ORANGE, shrinkA=1, shrinkB=1))
    ax.annotate("", xy=(89, 20.8), xytext=(89, 18), arrowprops=dict(kw, color=BLUE, shrinkA=1, shrinkB=1))
    ax.text(89, 22.5, "evaluated on the same B patients", ha="center", va="center", fontsize=6.6,
            bbox=dict(fc="white", ec=MUTED, lw=0.6, boxstyle="round,pad=0.3"))
    ax.text(76, 4.4, "Transport loss = local performance − external performance\n(paired patient-level bootstrap intervals)",
            ha="center", fontsize=6.6, color=MUTED, linespacing=1.3)
    save(fig, "Figure_1")


# ------------------------------------------------------------------ Figure 2
def fig2():
    fig = plt.figure(figsize=(7.1, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1], hspace=0.5, wspace=0.22)
    for i, T in enumerate(("B", "A")):
        r = load(T)
        ax = fig.add_subplot(gs[0, i])
        for j, arm in enumerate(cfg.ARMS):
            lo, ex = r["discrimination"][f"local:{arm}"], r["discrimination"][f"external:{arm}"]
            ax.plot([j - 0.12, j + 0.12], [lo["auroc"], ex["auroc"]], color=MUTED, lw=0.7, zorder=1)
            ax.errorbar(j - 0.12, lo["auroc"], yerr=[[lo["auroc"] - lo["auroc_ci"][0]], [lo["auroc_ci"][1] - lo["auroc"]]],
                        fmt="o", ms=5.5, color=BLUE, mfc=BLUE, mec="white", mew=0.8, elinewidth=1.1, capsize=0,
                        label="Local model" if j == 0 else None, zorder=3)
            ax.errorbar(j + 0.12, ex["auroc"], yerr=[[ex["auroc"] - ex["auroc_ci"][0]], [ex["auroc_ci"][1] - ex["auroc"]]],
                        fmt="s", ms=5, color=ORANGE, mfc="white", mec=ORANGE, mew=1.3, elinewidth=1.1, capsize=0,
                        label="External model" if j == 0 else None, zorder=3)
            loss = r["transport_loss"][arm]["auroc"]["est"]
            ax.text(j, min(lo["auroc_ci"][0], ex["auroc_ci"][0]) - 0.006, f"loss {loss:.3f}", fontsize=6.8,
                    color=MUTED, ha="center", va="top")
        ax.set_xticks(range(4)); ax.set_xticklabels([ARM_LABEL[a] for a in cfg.ARMS])
        ax.set_xlim(-0.5, 3.5); ax.set_ylim(0.62, 0.89)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        ax.set_ylabel("AUROC (95% CI)" if i == 0 else "")
        other = "A" if T == "B" else "B"
        ax.set_title(f"Evaluated in system {T} (external model from {other})", loc="left")
        if i == 0:
            ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2)
        panel_label(ax, "ab"[i])
    ax = fig.add_subplot(gs[1, :])
    rows, ypos = [], []
    y = 0
    for T in ("B", "A"):
        r = load(T)
        for sc, lab in (("local", "local model"), ("external", "external model")):
            d = r["decomposition"][sc]
            rows.append((f"System {T}, {lab}", d["shapley_values"]["est"], d["shapley_process"]["est"],
                         d["process_share"]["est"], d["process_share"]["ci"]))
            ypos.append(y); y += 1
        y += 0.5
    for (lab, v, p, share, sci), yy in zip(rows, ypos):
        ax.barh(yy, v, height=0.62, color=AQUA, edgecolor="white", lw=1.2)
        ax.barh(yy, p, left=v, height=0.62, color=YELLOW, edgecolor="white", lw=1.2, hatch="////")
        ax.text(v / 2, yy, f"{v:.3f}", ha="center", va="center", fontsize=7, color=INK)
        ax.text(v + p / 2, yy, f"{p:.3f}", ha="center", va="center", fontsize=7, color=INK,
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
        ax.text(v + p + 0.002, yy, f"process share {share * 100:.0f}% ({sci[0] * 100:.0f} to {sci[1] * 100:.0f})",
                va="center", fontsize=7, color=MUTED)
    ax.set_yticks(ypos); ax.set_yticklabels([r_[0] for r_ in rows]); ax.invert_yaxis()
    ax.set_xlim(0, 0.175); ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.set_xlabel("Gain in AUROC over the covariates-only arm, attributed to each information source")
    ax.barh(np.nan, 0, color=AQUA, label="Physiological values")
    ax.barh(np.nan, 0, color=YELLOW, hatch="////", edgecolor="white", label="Measurement process")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, handlelength=1.4, columnspacing=1.2,
              borderaxespad=0.1)
    panel_label(ax, "c")
    save(fig, "Figure_2")


# ------------------------------------------------------------------ Figure 3
def fig3():
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.6), gridspec_kw=dict(hspace=0.42, wspace=0.25))
    style = {"local:VP": (BLUE, "-", "Local model"), "external:VP": (ORANGE, "--", "External model"),
             "pooled:VP": (AQUA, ":", "Two-site model")}
    for i, T in enumerate(("B", "A")):
        r = load(T)
        ax = axes[0, i]
        lim = (0.2, 50)
        ax.plot(lim, lim, color=MUTED, lw=0.7, ls=(0, (1, 2)), label="Ideal")
        for k, (col, ls, lab) in style.items():
            c = r["calibration"][k]["curve"]
            pred, obs, n = np.array(c["pred"]), np.array(c["obs"]), np.array(c["n"])
            cum = np.cumsum(n) / n.sum()
            keep = (cum >= 0.01) & (cum <= 0.999)
            ax.plot(pred[keep] * 100, obs[keep] * 100, color=col, ls=ls, lw=1.6, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
        ticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50]
        ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
        ax.set_yticks(ticks); ax.set_yticklabels([str(t) for t in ticks]); ax.minorticks_off()
        ax.grid(True); ax.set_axisbelow(True)
        ax.set_xlabel("Predicted probability (%, log scale)")
        ax.set_ylabel("Observed proportion (%, log scale)" if i == 0 else "")
        ax.set_title(f"Calibration, evaluated in system {T}", loc="left")
        if i == 0:
            ax.legend(loc="lower right", handlelength=2.2)
        panel_label(ax, "ab"[i])
        ax = axes[1, i]
        nb = r["net_benefit"]
        pt = np.array([x["pt"] for x in nb["local:VP"]]) * 100
        ax.axhline(0, color=MUTED, lw=0.7, ls=(0, (1, 2)), label="Alert no one")
        ax.plot(pt, [x["nb_all"] * 1000 for x in nb["local:VP"]], color=MUTED, lw=1.0, ls="-.", label="Alert everyone")
        for k, (col, ls, lab) in style.items():
            ax.plot(pt, [x["nb"] * 1000 for x in nb[k]], color=col, ls=ls, lw=1.6,
                    marker="o" if ls == "-" else ("s" if ls == "--" else "^"), ms=3.2, label=lab)
        ax.set_ylim(-3, 19); ax.set_xlim(0.3, 10.2)
        ax.grid(True); ax.set_axisbelow(True)
        ax.set_xlabel("Threshold probability (%)")
        ax.set_ylabel("Net benefit per 1,000 patient-hours" if i == 0 else "")
        ax.set_title(f"Decision curves, evaluated in system {T}", loc="left")
        if i == 0:
            ax.legend(loc="upper right", handlelength=2.2)
        panel_label(ax, "cd"[i])
    save(fig, "Figure_3")


# ------------------------------------------------------------------ Figure 4
def fig4():
    fig = plt.figure(figsize=(7.1, 5.5))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.3, height_ratios=[1, 1.05])
    scen = [("development_internal", "Development site\n(threshold set here)"),
            ("external_at_dev_threshold", "External site,\nsame threshold"),
            ("external_at_retuned_threshold", "External site,\nthreshold re-set locally"),
            ("local", "Local model\nat the external site")]
    for i, T in enumerate(("B", "A")):
        r = load(T)
        S = r["source"]
        ax = fig.add_subplot(gs[0, i])
        al = r["alerting"]["VP"]
        for j, (k, lab) in enumerate(scen):
            e = al[k]
            col = ORANGE if k.startswith("external") else BLUE
            ax.plot([e["sensitivity"] * 100, e["false_alert_prop"] * 100], [j, j], color=GRID, lw=2.2, zorder=1)
            ax.scatter(e["sensitivity"] * 100, j, s=42, color=col, zorder=3, edgecolor="white", lw=0.8)
            ax.scatter(e["false_alert_prop"] * 100, j, s=38, marker="s", facecolor="white", edgecolor=col, lw=1.4, zorder=3)
            ax.text(e["sensitivity"] * 100 + 2.2, j - 0.02, f"{e['sensitivity'] * 100:.0f}", va="center", fontsize=7)
            ax.text(e["false_alert_prop"] * 100 - 2.2, j - 0.02, f"{e['false_alert_prop'] * 100:.0f}", va="center",
                    ha="right", fontsize=7)
        ax.set_yticks(range(4)); ax.set_yticklabels([s[1] for s in scen] if i == 0 else [])
        ax.invert_yaxis(); ax.set_xlim(0, 100); ax.xaxis.grid(True); ax.set_axisbelow(True)
        ax.set_xlabel("Patients alerted (%)")
        ax.set_title(f"Model developed in {S}, applied in {T}", loc="left")
        if i == 0:
            ax.scatter([], [], s=42, color=MUTED, label="Septic patients alerted by onset (sensitivity)")
            ax.scatter([], [], s=38, marker="s", facecolor="white", edgecolor=MUTED, lw=1.4,
                       label="Non-septic patients alerted")
            ax.legend(loc="upper left", bbox_to_anchor=(-0.62, -0.26), ncol=2, columnspacing=1.5, handletextpad=0.3)
        panel_label(ax, "ab"[i])
    up = json.load(open(cfg.RES / "updating.json"))
    for i, (key, T) in enumerate((("AtoB", "B"), ("BtoA", "A"))):
        ax = fig.add_subplot(gs[1, i])
        d = up[key]
        ns = sorted(int(k) for k in d)
        med = np.array([d[str(n)]["after"]["ici"]["median"] for n in ns]) * 100
        lo = np.array([d[str(n)]["after"]["ici"]["lo"] for n in ns]) * 100
        hi = np.array([d[str(n)]["after"]["ici"]["hi"] for n in ns]) * 100
        before = d[str(ns[-1])]["before"]["ici"]["median"] * 100
        local = load(T)["calibration"]["local:VP"]["ici"] * 100
        ax.fill_between(ns, lo, hi, color=AQUA, alpha=0.18, lw=0)
        ax.plot(ns, med, color=AQUA, lw=1.6, marker="^", ms=4, label="External model after recalibration")
        ax.axhline(before, color=ORANGE, lw=1.4, ls="--", label="External model, not recalibrated")
        ax.axhline(local, color=BLUE, lw=1.4, ls="-", label="Local model")
        ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([f"{n:,}" for n in ns])
        ax.minorticks_off(); ax.set_ylim(0, 2.0); ax.yaxis.grid(True); ax.set_axisbelow(True)
        ax.set_xlabel("Local patients used for recalibration")
        ax.set_ylabel("Integrated calibration index\n(percentage points)" if i == 0 else "")
        ax.set_title(f"Recalibrating the external model in {T}", loc="left")
        if i == 0:
            ax.legend(loc="upper right", handlelength=2.2)
        panel_label(ax, "cd"[i])
    save(fig, "Figure_4")


# ------------------------------------------------------------------ Supplementary figures
def figS1():
    av = pd.read_csv(cfg.RES / "variable_availability.csv", index_col=0)
    av = av.sort_values("A_pct_patients")
    fig, ax = plt.subplots(figsize=(5.2, 6.4))
    y = np.arange(len(av))
    ax.hlines(y, av["A_pct_patients"], av["B_pct_patients"], color=GRID, lw=2)
    ax.scatter(av["A_pct_patients"], y, s=26, color=ORANGE, marker="s", facecolor="white", lw=1.3, label="System A", zorder=3)
    ax.scatter(av["B_pct_patients"], y, s=28, color=BLUE, label="System B", zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(av.index); ax.set_xlim(-2, 102)
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.set_xlabel("Patients with at least one measurement (%)")
    ax.legend(loc="lower right")
    save(fig, "Figure_S1")


def figS2():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True, gridspec_kw=dict(wspace=0.08))
    cols = {"C": MUTED, "V": AQUA, "P": YELLOW, "VP": INK}
    mk = {"C": "x", "V": "o", "P": "s", "VP": "D"}
    for i, T in enumerate(("B", "A")):
        r = load(T)["by_time_to_onset"]
        labs = list(r)
        ax = axes[i]
        for arm in cfg.ARMS:
            for sc, ls, fill in (("local", "-", True), ("external", "--", False)):
                yv = [r[l][f"{sc}:{arm}"]["auroc"] for l in labs]
                ax.plot(range(len(labs)), yv, color=cols[arm], ls=ls, lw=1.3, marker=mk[arm], ms=4.5,
                        mfc=cols[arm] if fill else "white",
                        label=f"{ARM_LABEL[arm].replace(chr(10), ' ')}, {sc}" if i == 0 else None)
        ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        ax.set_xlabel("Hours relative to sepsis onset")
        ax.set_title(f"Evaluated in system {T}", loc="left")
        if i == 0:
            ax.set_ylabel("AUROC (septic hours in window\nvs all non-septic hours)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.25), ncol=4, columnspacing=1.0, fontsize=6.8)
    save(fig, "Figure_S2")


def figS3():
    sh = json.load(open(cfg.RES / "shap.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.6), gridspec_kw=dict(wspace=0.75))
    fam_col = {"process": YELLOW, "value": AQUA, "covariate": MUTED}
    for i, dev in enumerate(("A", "B")):
        top = sh[f"model_{dev}_on_{dev}"]["top20"][:15][::-1]
        ax = axes[i]
        for j, t in enumerate(top):
            ax.barh(j, t["share"] * 100, color=fam_col[t["family"]], height=0.66,
                    hatch="////" if t["family"] == "process" else None, edgecolor="white", lw=0.8)
        ax.set_yticks(range(len(top))); ax.set_yticklabels([t["feature"].replace("_", " ") for t in top])
        ax.xaxis.grid(True); ax.set_axisbelow(True)
        ax.set_xlabel("Share of total mean |SHAP| (%)")
        fs = sh[f"model_{dev}_on_{dev}"]["family_share"]
        ax.set_title(f"Model developed in {dev}\nprocess {fs['process'] * 100:.0f}%, values {fs['value'] * 100:.0f}%, "
                     f"covariates {fs['covariate'] * 100:.0f}%", loc="left")
        if i == 0:
            for f_, lab in (("value", "Physiological value"), ("process", "Measurement process"), ("covariate", "Covariate")):
                ax.barh(np.nan, 0, color=fam_col[f_], hatch="////" if f_ == "process" else None, edgecolor="white", label=lab)
            ax.legend(loc="lower right")
    save(fig, "Figure_S3")


def figS4():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.9), sharey=True, gridspec_kw=dict(wspace=0.06))
    for i, T in enumerate(("B", "A")):
        sub = load(T)["subgroups"]
        names = list(sub)
        ax = axes[i]
        for j, k in enumerate(names):
            lo, ex = sub[k]["local"], sub[k]["external"]
            ax.errorbar(lo["auroc"], j - 0.14, xerr=[[lo["auroc"] - lo["auroc_ci"][0]], [lo["auroc_ci"][1] - lo["auroc"]]],
                        fmt="o", ms=4.5, color=BLUE, mec="white", mew=0.7, elinewidth=1.0,
                        label="Local model" if j == 0 else None)
            ax.errorbar(ex["auroc"], j + 0.14, xerr=[[ex["auroc"] - ex["auroc_ci"][0]], [ex["auroc_ci"][1] - ex["auroc"]]],
                        fmt="s", ms=4.2, color=ORANGE, mfc="white", mew=1.2, elinewidth=1.0,
                        label="External model" if j == 0 else None)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([n.replace("sex:", "Sex: ").replace("age:", "Age: ").replace("unit:", "Unit: ").replace(">=", "80 or older|").replace("80 or older|80", "80 or older") for n in names])
        ax.invert_yaxis(); ax.xaxis.grid(True); ax.set_axisbelow(True); ax.set_xlim(0.68, 0.91)
        ax.set_xlabel("AUROC (95% CI)"); ax.set_title(f"Evaluated in system {T}", loc="left")
        if i == 0:
            ax.legend(loc="lower left")
    save(fig, "Figure_S4")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "S1", "S2", "S3", "S4"]
    for w in which:
        globals()[f"fig{w}"]()
        print("figure", w, "done")
