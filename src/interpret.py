"""Lightweight, template-guided interpretation layer.

Converts the quantitative pipeline outputs (segment profiles, risk gradients,
feature importances) into short, clinician-readable natural-language summaries.
This is the optional, template-guided component described in the manuscript; it
is deliberately simple and is provided as an LLM-ready scaffold (the same prompts
can be passed to an LLM for richer phrasing).

Input : results/stats_results.json, results/ml_results.json
Output: results/interpretation.txt
"""
import os, json

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")


def main():
    st = json.load(open(os.path.join(RES, "stats_results.json")))
    ml = json.load(open(os.path.join(RES, "ml_results.json")))
    lines = []

    pin = st["readmit_by_priorinpatient"]
    lines.append(
        f"Risk driver: 30-day readmission rises from {pin.get('0', 0)*100:.1f}% for patients with no "
        f"prior inpatient visits to {pin.get('5', 0)*100:.1f}% for those with five or more, making prior "
        f"inpatient utilisation the leading actionable risk factor.")

    prof = st["cluster_profile"]; rd = st["cluster_readmit"]; sz = st["cluster_sizes"]
    order = sorted(rd, key=lambda c: rd[c])
    tags = ["lower-risk", "intermediate-risk", "higher-risk"]
    for tag, c in zip(tags, order):
        p = prof[c]
        lines.append(
            f"Segment {c} ({tag}, n={sz[c]:,}): mean age {p['age']:.0f}, length of stay {p['LOS']:.1f} days, "
            f"{p['meds']:.0f} medications, {p['diagnoses']:.0f} diagnoses; 30-day readmission {rd[c]*100:.1f}%.")

    best = max(ml["res"]["models"].items(), key=lambda kv: kv[1]["roc_auc"])
    lines.append(
        f"Best-discriminating model: {best[0]} (ROC-AUC {best[1]['roc_auc']:.3f}, recall {best[1]['recall']*100:.0f}%). "
        f"Note that higher-accuracy models are misleading under class imbalance.")

    top = ", ".join(n.replace("_", " ") for n, _ in ml["res"]["rf_top_features"][:5])
    lines.append(f"Most influential variables (Random Forest): {top}.")

    out = os.path.join(RES, "interpretation.txt")
    open(out, "w").write("\n\n".join(lines) + "\n")
    print("Saved interpretation.txt\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
