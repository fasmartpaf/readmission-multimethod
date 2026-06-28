"""Supervised modelling and imbalance-aware evaluation, plus the cluster-augmentation test.

Input : results/X.npy, results/y.npy, results/cluster_labels.npy, results/feat_names.json
Output: results/ml_results.json

Models: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbours.
Metrics: accuracy, precision, recall, F1 (positive class), ROC-AUC, PR-AUC, confusion matrix.
Novelty test: does adding K-Means segment membership improve 5-fold CV ROC-AUC?
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve)

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
SEED = 42


def main():
    X = np.load(os.path.join(RES, "X.npy"))
    y = np.load(os.path.join(RES, "y.npy"))
    clusters = np.load(os.path.join(RES, "cluster_labels.npy"))
    feat_names = json.load(open(os.path.join(RES, "feat_names.json")))

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    res = {"n_train": int(len(ytr)), "n_test": int(len(yte)), "pos_rate": round(float(y.mean()), 4), "models": {}}
    curves = {}

    def evaluate(name, model, Xa, Xb, ytr_):
        t = time.time(); model.fit(Xa, ytr_)
        p = model.predict(Xb); s = model.predict_proba(Xb)[:, 1]
        res["models"][name] = {"accuracy": round(accuracy_score(yte, p), 4),
            "precision": round(precision_score(yte, p, zero_division=0), 4),
            "recall": round(recall_score(yte, p, zero_division=0), 4),
            "f1": round(f1_score(yte, p, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(yte, s), 4),
            "pr_auc": round(average_precision_score(yte, s), 4),
            "cm": confusion_matrix(yte, p).tolist()}
        fpr, tpr, _ = roc_curve(yte, s); step = max(1, len(fpr) // 200)
        curves[name] = {"fpr": list(np.round(fpr[::step], 4)), "tpr": list(np.round(tpr[::step], 4))}
        print(f"{name}: acc={res['models'][name]['accuracy']} AUC={res['models'][name]['roc_auc']} "
              f"F1={res['models'][name]['f1']} ({time.time()-t:.1f}s)")
        return model

    evaluate("Logistic Regression", LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=-1), Xtr_s, Xte_s, ytr)
    evaluate("Decision Tree", DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=SEED), Xtr, Xte, ytr)
    rf = evaluate("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=16, class_weight="balanced", n_jobs=-1, random_state=SEED), Xtr, Xte, ytr)
    # KNN on stratified subsample for tractability (high-dimensional distances)
    idx = pd.Series(range(len(ytr))).groupby(ytr).sample(frac=min(1.0, 30000 / len(ytr)), random_state=1).values
    evaluate("KNN", KNeighborsClassifier(n_neighbors=25, n_jobs=-1), Xtr_s[idx], Xte_s, ytr[idx])
    res["knn_train_n"] = int(len(idx))

    imp = sorted(zip(feat_names, rf.feature_importances_), key=lambda x: -x[1])[:12]
    res["rf_top_features"] = [[n, round(float(v), 4)] for n, v in imp]

    # Novelty: cluster-augmented vs baseline (5-fold CV ROC-AUC)
    Xaug = np.hstack([X, np.eye(3)[clusters]]).astype(np.float32)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    base = RandomForestClassifier(n_estimators=150, max_depth=16, class_weight="balanced", n_jobs=-1, random_state=SEED)
    ab = cross_val_score(base, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    aa = cross_val_score(base, Xaug, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    res["cv_auc_baseline"] = [round(float(ab.mean()), 4), round(float(ab.std()), 4)]
    res["cv_auc_cluster_aug"] = [round(float(aa.mean()), 4), round(float(aa.std()), 4)]
    print("CV AUC baseline", res["cv_auc_baseline"], "| cluster-augmented", res["cv_auc_cluster_aug"])

    json.dump({"res": res, "curves": curves}, open(os.path.join(RES, "ml_results.json"), "w"), indent=1)
    print("Saved ml_results.json")


if __name__ == "__main__":
    main()
