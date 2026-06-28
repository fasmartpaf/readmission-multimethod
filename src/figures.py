"""Generate all manuscript figures from saved results.

Input : results/stats_results.json, results/ml_results.json, results/clean.csv, results/cluster_labels.npy
Output: figures/Figure 1.png ... Figure 9.png
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")

NAVY, BLUE, TEAL, SEAFOAM, GREY, LIGHT, RED = "#0B2545", "#1B6CA8", "#0D9488", "#2EC4B6", "#64748B", "#E2E8F0", "#B00020"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#CBD5E1", "text.color": "#1E293B",
    "axes.labelcolor": "#334155", "xtick.color": "#475569", "ytick.color": "#475569"})


def style(ax, title, yl=None, xl=None):
    ax.set_title(title, fontsize=13, weight="bold", color=NAVY, pad=10)
    if yl: ax.set_ylabel(yl, fontsize=11, weight="bold")
    if xl: ax.set_xlabel(xl, fontsize=11, weight="bold")
    ax.yaxis.grid(True, color=LIGHT, zorder=0); ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)


def save(name): plt.tight_layout(); plt.savefig(os.path.join(FIG, name), bbox_inches="tight", dpi=200); plt.close()


def main():
    os.makedirs(FIG, exist_ok=True)
    st = json.load(open(os.path.join(RES, "stats_results.json")))
    ml = json.load(open(os.path.join(RES, "ml_results.json")))
    d = pd.read_csv(os.path.join(RES, "clean.csv"), low_memory=False)

    # Figure 1: distributions
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    axs[0].hist(d["time_in_hospital"], bins=14, color=TEAL, zorder=3, edgecolor="white"); style(axs[0], "Time in Hospital", "Count", "Days")
    axs[1].hist(d["num_medications"], bins=30, color=BLUE, zorder=3, edgecolor="white"); style(axs[1], "Number of Medications", "Count", "Medications")
    save("Figure 1.png")

    # Figure 2: readmission by prior inpatient
    g = st["readmit_by_priorinpatient"]; xs = list(g); ys = [g[k]*100 for k in xs]; xl = [x if x != "5" else "5+" for x in xs]
    fig, ax = plt.subplots(figsize=(7, 4.3)); b = ax.bar(xl, ys, color=TEAL, zorder=3, width=0.6)
    for bi, v in zip(b, ys): ax.text(bi.get_x()+bi.get_width()/2, v+0.5, f"{v:.1f}%", ha="center", weight="bold", color=NAVY, fontsize=10)
    style(ax, "30-Day Readmission Rate by Prior Inpatient Visits", "Readmission rate (%)", "Number of prior inpatient visits"); ax.set_ylim(0, 40); save("Figure 2.png")

    # Figure 3: readmission by age
    g = st["readmit_by_age"]; xs = sorted(g, key=lambda k: int(k)); ys = [g[k]*100 for k in xs]
    fig, ax = plt.subplots(figsize=(7, 4.3)); ax.plot([int(x) for x in xs], ys, marker="o", color=BLUE, lw=2.5, ms=7, zorder=3)
    style(ax, "30-Day Readmission Rate by Age", "Readmission rate (%)", "Age (years, bracket midpoint)"); save("Figure 3.png")

    # Figure 4: correlation with target
    ct = sorted(st["corr_with_target"].items(), key=lambda x: x[1])
    labels = [k.replace("_", " ") for k, _ in ct]; vals = [v for _, v in ct]
    fig, ax = plt.subplots(figsize=(7, 4.3)); ax.barh(labels, vals, color=[TEAL if v >= 0 else GREY for v in vals], zorder=3)
    ax.axvline(0, color="#94A3B8", lw=0.8); style(ax, "Correlation of Features with 30-Day Readmission", None, "Pearson r")
    for i, v in enumerate(vals): ax.text(v+(0.002 if v >= 0 else -0.002), i, f"{v:.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=9, color=NAVY)
    ax.set_xlim(-0.02, 0.12); save("Figure 4.png")

    m = ml["res"]["models"]; names = ["Logistic Regression", "Decision Tree", "Random Forest", "KNN"]
    # Figure 5: model comparison
    acc = [m[n]["accuracy"] for n in names]; auc = [m[n]["roc_auc"] for n in names]; f1 = [m[n]["f1"] for n in names]
    x = np.arange(len(names)); w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x-w, acc, w, label="Accuracy", color=GREY, zorder=3); ax.bar(x, auc, w, label="ROC-AUC", color=TEAL, zorder=3); ax.bar(x+w, f1, w, label="F1 (readmit)", color=BLUE, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(["Logistic\nRegression", "Decision\nTree", "Random\nForest", "KNN"], fontsize=10)
    ax.axhline(0.5, color=RED, lw=1, ls="--"); ax.text(3.3, 0.51, "AUC chance", color=RED, fontsize=8)
    style(ax, "Model Performance: Accuracy is Misleading under Imbalance", "Score"); ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center"); save("Figure 5.png")

    # Figure 6: ROC
    cv = ml["curves"]; cols = {"Logistic Regression": TEAL, "Decision Tree": BLUE, "Random Forest": NAVY, "KNN": GREY}
    fig, ax = plt.subplots(figsize=(6, 5.2))
    for n in names: ax.plot(cv[n]["fpr"], cv[n]["tpr"], label=f"{n} (AUC {m[n]['roc_auc']:.3f})", color=cols[n], lw=2.2)
    ax.plot([0, 1], [0, 1], ls="--", color="#94A3B8", lw=1); style(ax, "ROC Curves — 30-Day Readmission", "True positive rate", "False positive rate")
    ax.legend(frameon=False, fontsize=9, loc="lower right"); save("Figure 6.png")

    # Figure 7: feature importance
    fi = ml["res"]["rf_top_features"][::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6)); ax.barh([n.replace("_", " ")[:26] for n, _ in fi], [v for _, v in fi], color=BLUE, zorder=3)
    style(ax, "Random Forest Feature Importance (Top 12)", None, "Importance"); save("Figure 7.png")

    # Figure 8: confusion matrix (LR)
    cm = np.array(m["Logistic Regression"]["cm"])
    fig, ax = plt.subplots(figsize=(4.6, 4.2)); im = ax.imshow(cm, cmap="BuGn")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=13, weight="bold", color="white" if cm[i, j] > cm.max()/2 else NAVY)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["No", "Readmit"]); ax.set_yticklabels(["No", "Readmit"])
    ax.set_xlabel("Predicted", weight="bold"); ax.set_ylabel("Actual", weight="bold")
    ax.set_title("Confusion Matrix — Logistic Regression", fontsize=12, weight="bold", color=NAVY, pad=10); save("Figure 8.png")

    # Figure 9: cluster readmission + sizes
    cr = st["cluster_readmit"]; cs = st["cluster_sizes"]; ks = sorted(cr)
    fig, ax = plt.subplots(figsize=(7, 4.3)); b = ax.bar([f"Cluster {k}" for k in ks], [cr[k]*100 for k in ks], color=[BLUE, TEAL, SEAFOAM], zorder=3, width=0.55)
    for bi, k in zip(b, ks): ax.text(bi.get_x()+bi.get_width()/2, cr[k]*100+0.15, f"{cr[k]*100:.1f}%\n(n={cs[k]:,})", ha="center", weight="bold", color=NAVY, fontsize=9)
    style(ax, "Readmission Rate by Patient Segment (K-Means, k=3)", "Readmission rate (%)"); ax.set_ylim(0, 14); save("Figure 9.png")

    print(f"Saved 9 figures to {FIG}")


if __name__ == "__main__":
    main()
