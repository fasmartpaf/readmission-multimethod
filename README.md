# Multi-Method Analysis of 30-Day Hospital Readmission

Reproducible code for the paper *"Multi-Method Analysis Based on Modeling, Machine
Learning and Patient Segmentation: A Reproducible, Interpretable Study of 30-Day
Hospital Readmission."*

The pipeline integrates **descriptive statistics**, **supervised machine learning**,
**unsupervised patient segmentation (K-Means)**, and a **lightweight, template-guided
interpretation layer**, with an explicit, imbalance-aware evaluation.

## Dataset

UCI Machine Learning Repository — *Diabetes 130-US Hospitals for Years 1999–2008*.
- DOI: 10.24432/C5230J
- Origin paper: Strack et al. (2014), *BioMed Research International*, DOI: 10.1155/2014/781670
- 101,766 encounters; after excluding death/hospice discharges and keeping the first
  encounter per patient, **69,990 patients** remain. Outcome: **30-day readmission** (~9.0% positive).

The dataset is public and is downloaded automatically by `src/download_data.py`.

## Requirements

- Python 3.9+
- See `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Reproduce everything

```bash
cd src
python run_all.py
```

This runs all six steps and writes outputs to `results/` and `figures/`.
Runtime is a few minutes on a standard laptop. All random seeds are fixed (42), so
results are deterministic.

### Or run steps individually (from `src/`)

```bash
python download_data.py     # -> data/diabetic_data.csv
python prep.py              # -> results/clean.csv, X.npy, y.npy, feat_names.json
python stats_cluster.py     # -> results/stats_results.json, cluster_labels.npy
python models.py            # -> results/ml_results.json
python figures.py           # -> figures/Figure 1.png ... Figure 9.png
python interpret.py         # -> results/interpretation.txt
```

## Outputs and where they appear in the paper

| Output | Paper element |
|---|---|
| `results/stats_results.json` | Section 4 descriptive stats, correlations, risk strata, segments |
| `results/ml_results.json` | Table 1; Figures 5–8 |
| `figures/Figure 1–9.png` | Figures 1–9 |
| `results/interpretation.txt` | Section 4 interpretation-layer example |

### Headline results (deterministic, seed = 42)

- Best discrimination: **Logistic Regression, ROC-AUC ≈ 0.652** (recall ≈ 0.52).
- Accuracy paradox: Random Forest (acc ≈ 0.85) and KNN (acc ≈ 0.91) have F1 ≈ 0.18 and 0.01.
- Prior inpatient visits: readmission rises from ≈ 8% to ≈ 36%.
- Three K-Means segments: readmission ≈ 6.6% / 9.7% / 11.1%.
- Cluster-augmented vs baseline 5-fold CV ROC-AUC: ≈ 0.618 vs 0.611 (no predictive gain).

> Minor numerical variation (±0.005) across library versions is expected; the
> qualitative conclusions are stable.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── src/
    ├── download_data.py
    ├── prep.py
    ├── stats_cluster.py
    ├── models.py
    ├── figures.py
    ├── interpret.py
    └── run_all.py
```
(`data/`, `results/`, and `figures/` are created at runtime.)

## License

Code released under the MIT License (see `LICENSE`). The dataset is distributed by UCI
under its own terms; please cite Strack et al. (2014) and the UCI repository.

## How to cite

If you use this code, please cite the paper and this repository (see `CITATION.cff`).
