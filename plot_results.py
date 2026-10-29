"""Turn results/results.csv into figures and a plain-language summary."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TITLES = {
    "missing_random": "Missing at random",
    "missing_group": "Missing in minority group",
    "label_noise": "Random label noise",
    "label_noise_group": "Minority positives mislabeled",
    "label_noise_matched": "Random noise, same # of flips",
    "selection_bias": "Minority positives under-sampled",
}
COLORS = {"logistic / mean": "#4C72B0", "logistic / mean+indicator": "#55A868",
          "logistic / drop_rows": "#C44E52", "boosting / native": "#DD8452"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results/results.csv")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    out = Path(args.outdir)
    res = pd.read_csv(args.inp)
    dataset = res["dataset"].iloc[0]
    agg = (res.groupby(["corruption", "severity", "setup"])[["accuracy", "tpr_gap", "selection_gap", "n_train", "share_changed"]]
              .agg(["mean", "std"]).reset_index())

    for metric, ylabel, fname in [
        ("accuracy", "Accuracy on clean test set", "fig_accuracy.png"),
        ("tpr_gap", "TPR gap (majority − minority)", "fig_tpr_gap.png"),
    ]:
        fig, axes = plt.subplots(1, len(TITLES), figsize=(20, 3.6), sharey=True)
        for ax, kind in zip(axes, TITLES):
            sub = agg[agg["corruption"] == kind]
            for setup, d in sub.groupby("setup"):
                m, s = d[(metric, "mean")], d[(metric, "std")].fillna(0)
                ax.plot(d["severity"], m, marker="o", ms=3, label=setup, color=COLORS.get(setup))
                ax.fill_between(d["severity"], m - s, m + s, alpha=0.15, color=COLORS.get(setup))
            if metric == "tpr_gap":
                ax.axhline(0, color="grey", lw=0.8, ls="--")
            ax.set_title(TITLES[kind], fontsize=10)
            ax.set_xlabel("Severity")
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(f"{ylabel} as training data gets worse — {dataset}", y=1.02)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(out / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Summary: change from clean (severity 0) to the worst level, per corruption and setup
    m = res.groupby(["corruption", "severity", "setup"])[["accuracy", "tpr_gap", "share_changed"]].mean().reset_index()
    clean = m[m["severity"] == 0].set_index(["corruption", "setup"])
    worst = m[m["severity"] == m["severity"].max()].set_index(["corruption", "setup"])
    change = pd.DataFrame({
        "acc_clean": clean["accuracy"], "acc_worst": worst["accuracy"],
        "tpr_gap_clean": clean["tpr_gap"], "tpr_gap_worst": worst["tpr_gap"],
        "share_changed_worst": worst["share_changed"],
    }).dropna()
    change["acc_change"] = change["acc_worst"] - change["acc_clean"]
    change["gap_change"] = change["tpr_gap_worst"] - change["tpr_gap_clean"]
    change = change.round(3).reset_index()
    change.to_csv(out / "summary_table.csv", index=False)

    biggest_acc = change.loc[change["acc_change"].idxmin()]
    biggest_gap = change.loc[change["gap_change"].idxmax()]
    lines = [
        f"# Results: {dataset}",
        "",
        f"Seeds: {res['seed'].nunique()} | Severities: {', '.join(f'{s:g}' for s in sorted(res['severity'].unique()))}",
        "",
        "## Change from clean training data to the worst severity",
        change.to_markdown(index=False),
        "",
        "## Headline",
        f"- Largest accuracy drop: **{TITLES[biggest_acc['corruption']]}** with {biggest_acc['setup']} "
        f"({biggest_acc['acc_change']:+.3f}).",
        f"- Largest fairness harm (TPR gap increase): **{TITLES[biggest_gap['corruption']]}** with "
        f"{biggest_gap['setup']} ({biggest_gap['gap_change']:+.3f}).",
        "- Compare the two: corruptions that barely move accuracy can still widen the gap between groups.",
        "",
        "![Accuracy](fig_accuracy.png)",
        "",
        "![TPR gap](fig_tpr_gap.png)",
    ]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines[:-4]))


if __name__ == "__main__":
    main()
