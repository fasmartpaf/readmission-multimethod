"""Descriptive statistics, correlations, risk stratification, and K-Means segmentation.

Input : results/clean.csv
Output: results/stats_results.json, results/cluster_labels.npy
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from prep import NUMERIC

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")


def main(k=3, seed=42):
    d = pd.read_csv(os.path.join(RES, "clean.csv"), low_memory=False)
    res = {"n": int(len(d)), "pos_rate": round(float(d["readmit30"].mean()), 4)}

    labels = {'age_ord': 'Age', 'time_in_hospital': 'Time in hospital (days)',
              'num_medications': 'Num medications', 'num_lab_procedures': 'Num lab procedures',
              'number_diagnoses': 'Num diagnoses', 'number_inpatient': 'Prior inpatient visits',
              'number_emergency': 'Prior emergency visits'}
    res["descriptive"] = {name: {"mean": round(d[c].mean(), 2), "median": float(d[c].median()),
                                  "std": round(d[c].std(), 2), "min": float(d[c].min()), "max": float(d[c].max())}
                          for c, name in labels.items()}

    cm = d[NUMERIC + ["readmit30"]].corr()
    res["corr_with_target"] = {k_: round(v, 3) for k_, v in cm["readmit30"].drop("readmit30").sort_values().items()}
    res["corr_matrix"] = {a: {b: round(cm.loc[a, b], 3) for b in NUMERIC} for a in NUMERIC}

    res["readmit_by_priorinpatient"] = {str(k_): round(v, 3) for k_, v in
        d.groupby(d["number_inpatient"].clip(upper=5))["readmit30"].mean().items()}
    res["readmit_by_age"] = {str(int(k_)): round(v, 3) for k_, v in
        d.groupby("age_ord")["readmit30"].mean().items()}

    Xc = StandardScaler().fit_transform(d[NUMERIC].fillna(0))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Xc)
    d["cluster"] = km.labels_
    res["cluster_sizes"] = {str(k_): int(v) for k_, v in pd.Series(km.labels_).value_counts().sort_index().items()}
    res["cluster_readmit"] = {str(k_): round(v, 3) for k_, v in d.groupby("cluster")["readmit30"].mean().items()}
    res["cluster_profile"] = {}
    for cl in sorted(d["cluster"].unique()):
        sub = d[d["cluster"] == cl]
        res["cluster_profile"][str(cl)] = {n: round(sub[c].mean(), 2) for c, n in
            [('age_ord', 'age'), ('time_in_hospital', 'LOS'), ('num_medications', 'meds'),
             ('number_inpatient', 'prior_inpatient'), ('number_diagnoses', 'diagnoses')]}

    np.save(os.path.join(RES, "cluster_labels.npy"), km.labels_)
    json.dump(res, open(os.path.join(RES, "stats_results.json"), "w"), indent=1)
    print("Saved stats_results.json, cluster_labels.npy")
    print("Correlation with target:", res["corr_with_target"])
    print("Cluster readmission rates:", res["cluster_readmit"])


if __name__ == "__main__":
    main()
