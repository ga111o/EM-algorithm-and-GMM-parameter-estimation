"""Figures (figures/*.pdf, *.png) and LaTeX tables (results/tables/*.tex) from saved results.

Usage: uv run python report_assets.py   (after run_small.py)
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_small import CFG, DATA, PAR, RES, log_density

ROOT = Path(__file__).resolve().parent
FIG, TAB = ROOT / "figures", RES / "tables"
CASE = CFG["figure_case"]
COLORS = ["#215A8A", "#C36624", "#287D68"]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def em_curves():
    cond, seed = CASE["condition"], CASE["seed"]
    hist = [h for h in read_csv(RES / "em_history.csv") if h["condition"] == cond and int(h["seed"]) == seed]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    n_nonpos, finals = 0, []
    for r in range(CFG["n_restarts"]):
        h = [x for x in hist if int(x["restart"]) == r]
        it = np.array([int(x["iteration"]) for x in h])
        ll = np.array([float(x["loglik"]) for x in h])
        dl = np.array([float(x["delta"]) for x in h])
        ax1.plot(it, ll, color=COLORS[r], lw=1.4, label=f"restart {r}")
        finals.append(ll[-1])
        pos = np.isfinite(dl) & (dl > 0)
        n_nonpos += int((np.isfinite(dl) & (dl <= 0)).sum())
        ax2.plot(it[pos], dl[pos], color=COLORS[r], lw=1.2, label=f"restart {r}")
    ax1.set(xlabel="EM update $t$", ylabel=r"train log-likelihood $\ell_t$ (sum)",
            ylim=(min(finals) - 25, max(finals) + 3))  # zoom near final values; early iterates fall below
    ax2.set(xlabel="EM update $t$", ylabel=r"$\Delta\ell_t=\ell_t-\ell_{t-1}$", yscale="log")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "em_curves")
    return n_nonpos


def density_figure():
    cond, seed = CASE["condition"], CASE["seed"]
    d = np.load(DATA / f"{cond}_seed{seed}.npz")
    runs = [r for r in read_csv(RES / "metrics_by_run.csv")
            if r["condition"] == cond and int(r["seed"]) == seed and r["method"] == "EM"
            and r["status"] != "numerical_failure"]
    best = max(runs, key=lambda r: float(r["train_loglik"]))
    worst = min(runs, key=lambda r: float(r["train_loglik"]))
    lo, hi = d["box_lo"], d["box_hi"]
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 300), np.linspace(lo[1], hi[1], 300))
    G = np.c_[gx.ravel(), gy.ravel()]
    levels = np.linspace(-12, 0, 13)
    panels = [("True density $p_*$", (d["pi"], d["mu"], d["Sigma"]), None)]
    for tag, r in (("highest", best), ("lowest", worst)):
        p = np.load(PAR / f"{cond}_seed{seed}_em_r{r['restart']}.npz")
        panels.append((f"EM restart {r['restart']} ({tag} final $\\ell$)", (p["pi"], p["mu"], p["Sigma"]), p["mu0"]))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharex=True, sharey=True, layout="constrained")
    nn, na = 300, 100
    for ax, (title, params, mu0) in zip(axes, panels):
        z = log_density(G, *params).reshape(gx.shape)
        cs = ax.contourf(gx, gy, np.clip(z, levels[0], None), levels=levels, cmap="Blues")
        ax.scatter(*d["Xte"][:nn].T, s=3, c="0.35", lw=0, label=f"normal test (first {nn})")
        ax.scatter(*d["Xan"][:na].T, s=6, c="#C36624", marker="x", lw=0.7, label=f"anomaly (first {na})")
        ax.scatter(*params[1].T, s=40, c="k", marker="o", label="final mean" if mu0 is not None else "true mean")
        if mu0 is not None:
            ax.scatter(*mu0.T, s=50, facecolors="none", edgecolors="r", marker="s", label="initial mean")
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
    fig.legend(*axes[2].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=8, frameon=False)
    fig.colorbar(cs, ax=axes, shrink=0.8, label=r"$\log p(x)$ (clipped at $-12$)")
    save(fig, "density")
    return best, worst


def main_figure():
    rows = read_csv(RES / "metrics_by_seed.csv")
    metrics = (("E_pi", r"$E_\pi$"), ("E_mu", r"$E_\mu$"), ("E_Sigma", r"$E_\Sigma$"), ("test_nll", "Test NLL"))
    methods = (("EM", "EM", "o", -0.15), ("DSM", "Parametric DSM", "s", 0.15))
    conds = ("a1", "a4")
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.6), layout="constrained")
    for ax, (m, mlab) in zip(axes, metrics):
        vals = {(c, meth): np.array([float(r[m]) for r in sorted(rows, key=lambda r: int(r["seed"]))
                                     if r["condition"] == c and r["method"] == meth])
                for c in conds for meth, *_ in methods}
        for i, c in enumerate(conds):
            # same seed (same data and initial values) joined by a line
            for em, dsm in zip(vals[(c, "EM")], vals[(c, "DSM")]):
                ax.plot([i + methods[0][3], i + methods[1][3]], [em, dsm], color="0.8", lw=0.7, zorder=1)
            for (meth, label, marker, dx), color in zip(methods, COLORS):
                v = vals[(c, meth)]
                ax.scatter(np.full(v.size, i + dx), v, s=14, c=color, marker=marker, alpha=0.45, lw=0, zorder=2)
                ax.errorbar(i + dx + 0.09 * np.sign(dx), v.mean(), yerr=v.std(ddof=1), fmt=marker, ms=6,
                            color=color, capsize=3, lw=1.3, zorder=3, label=label if i == 0 else None)
        ax.set_xticks(range(len(conds)), ["$a=1$", "$a=4$"])
        ax.set_xlim(-0.6, len(conds) - 0.4)
        if m != "test_nll":
            ax.set_ylim(bottom=0)  # RMSE >= 0; mean - sd can fall below it
        ax.set_title(mlab, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=9, frameon=False)
    save(fig, "main_comparison")


def pm(mean, sd, nd=3):
    return f"${mean:.{nd}f}\\pm{sd:.{nd}f}$"


def tables():
    TAB.mkdir(parents=True, exist_ok=True)
    s = {(r["table"], r["condition"], r["method"], r["metric"]): (float(r["mean"]), float(r["sd"]), int(r["n_seeds"]))
         for r in read_csv(RES / "summary.csv")}

    def row(cond, method, metrics, table="selected_model", nd=None):
        nd = nd or {}
        return " & ".join(pm(*s[(table, cond, method, m)][:2], nd.get(m, 3)) for m in metrics)

    pm_cols = ("E_pi", "E_mu", "E_Sigma", "test_nll")
    main = [f"EM & {row('a1', 'EM', pm_cols)} \\\\", f"Parametric DSM & {row('a1', 'DSM', pm_cols)} \\\\"]
    (TAB / "tab_main.tex").write_text("\n".join(main) + "\n")

    labels = {"a4": "$a=4$", "a1_rare": "$a=1$, rare $\\pi$", "a1_single": "$a=1$, $t\\in\\{0.2\\}$"}
    cond_rows = []
    for c in ("a4", "a1_rare", "a1_single"):
        for m, mname in (("EM", "EM"), ("DSM", "Parametric DSM")):
            if ("selected_model", c, m, "E_pi") in s:
                cond_rows.append(f"{labels[c]} & {mname} & {row(c, m, pm_cols)} \\\\")
    (TAB / "tab_conditions.tex").write_text("\n".join(cond_rows) + "\n")

    det_cols = ("test_fpr", "recall", "precision", "auroc", "pr_auc_trapz")
    mlabel = {"em_best_train": "EM, highest $\\ell$", "em_worst_train": "EM, lowest $\\ell$",
              "em_selected": "EM, selected", "dsm_selected": "DSM, selected"}
    det_rows = []
    for c in ("a1", "a4", "a1_rare", "a1_single"):
        # highest/lowest train-likelihood restarts only for the main a=1 case; all rows are in summary.csv
        for m in mlabel if c == "a1" else ("em_selected", "dsm_selected"):
            if ("detection", c, m, "auroc") in s:
                lab = labels.get(c, "$a=1$")
                det_rows.append(f"{lab} & {mlabel[m]} & "
                                f"{row(c, m, det_cols, 'detection', {'auroc': 4, 'pr_auc_trapz': 4})} \\\\")
    (TAB / "tab_detection.tex").write_text("\n".join(det_rows) + "\n")


def main():
    FIG.mkdir(exist_ok=True)
    n_nonpos = em_curves()
    best, worst = density_figure()
    main_figure()
    tables()
    print(json.dumps(dict(em_curves_nonpositive_delta=n_nonpos, best_restart=best["restart"],
                          worst_restart=worst["restart"]), indent=2))


if __name__ == "__main__":
    main()
