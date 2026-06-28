"""Cohort construction, preprocessing, and feature engineering.

Input : data/diabetic_data.csv
Output: results/clean.csv, results/X.npy, results/y.npy, results/feat_names.json

Decisions (documented for reproducibility):
- Target: 30-day readmission (readmitted == '<30') vs. otherwise.
- Exclude discharges to death/hospice (cannot be readmitted).
- Keep one (first) encounter per patient to avoid label leakage.
- ICD-9 diagnoses grouped into clinical categories.
- Drug change variables encoded ordinally (No/Down/Steady/Up).
- Age brackets mapped to bracket midpoints.
"""
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "diabetic_data.csv")
RES = os.path.join(HERE, "..", "results")

DRUGS = ['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide',
 'glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol',
 'troglitazone','tolazamide','examide','citoglipton','insulin','glyburide-metformin',
 'glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone']

NUMERIC = ['age_ord','time_in_hospital','num_lab_procedures','num_procedures','num_medications',
 'number_outpatient','number_emergency','number_inpatient','number_diagnoses']


def icd9_group(code):
    if pd.isna(code):
        return 'Missing'
    s = str(code)
    if s.startswith('V') or s.startswith('E'):
        return 'Other'
    try:
        iv = int(float(s))
    except ValueError:
        return 'Other'
    if s.startswith('250'): return 'Diabetes'
    if 390 <= iv <= 459 or iv == 785: return 'Circulatory'
    if 460 <= iv <= 519 or iv == 786: return 'Respiratory'
    if 520 <= iv <= 579 or iv == 787: return 'Digestive'
    if 580 <= iv <= 629 or iv == 788: return 'Genitourinary'
    if 800 <= iv <= 999: return 'Injury'
    if 710 <= iv <= 739: return 'Musculoskeletal'
    if 140 <= iv <= 239: return 'Neoplasms'
    return 'Other'


def load():
    d = pd.read_csv(DATA, na_values='?', low_memory=False)
    d = d[~d['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])].copy()
    d = d.sort_values('encounter_id').drop_duplicates('patient_nbr', keep='first').copy()
    d['readmit30'] = (d['readmitted'] == '<30').astype(int)
    age_map = {f'[{i}-{i+10})': i + 5 for i in range(0, 100, 10)}
    d['age_ord'] = d['age'].map(age_map)
    d['race'] = d['race'].fillna('Missing')
    for c in ['diag_1', 'diag_2', 'diag_3']:
        d[c + '_grp'] = d[c].apply(icd9_group)
    dose = {'No': 0, 'Down': 1, 'Steady': 2, 'Up': 3}
    keep_drugs = [c for c in DRUGS if d[c].nunique() > 1]
    for c in keep_drugs:
        d[c + '_ord'] = d[c].map(dose).fillna(0)
    d['change_b'] = (d['change'] == 'Ch').astype(int)
    d['diabetesMed_b'] = (d['diabetesMed'] == 'Yes').astype(int)
    d['max_glu_ord'] = d['max_glu_serum'].map({'None': 0, 'Norm': 1, '>200': 2, '>300': 3}).fillna(0)
    d['A1C_ord'] = d['A1Cresult'].map({'None': 0, 'Norm': 1, '>7': 2, '>8': 3}).fillna(0)
    return d, keep_drugs


def build_matrix(d, keep_drugs):
    cat = ['race', 'gender', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
           'diag_1_grp', 'diag_2_grp', 'diag_3_grp']
    d['gender'] = d['gender'].replace('Unknown/Invalid', np.nan)
    base = d[NUMERIC].copy()
    drug_ord = d[[c + '_ord' for c in keep_drugs]].copy()
    extra = d[['change_b', 'diabetesMed_b', 'max_glu_ord', 'A1C_ord']].copy()
    dummies = pd.get_dummies(d[cat].astype(str), drop_first=True)
    X = pd.concat([base, drug_ord, extra, dummies], axis=1)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    return X, d['readmit30'].values


def main():
    os.makedirs(RES, exist_ok=True)
    d, kd = load()
    X, y = build_matrix(d, kd)
    print(f"Cohort: {len(d):,} patients | 30-day readmission rate: {y.mean():.4f} | features: {X.shape[1]}")
    d.to_csv(os.path.join(RES, "clean.csv"), index=False)
    np.save(os.path.join(RES, "X.npy"), X.values.astype(np.float32))
    np.save(os.path.join(RES, "y.npy"), y)
    json.dump(list(X.columns), open(os.path.join(RES, "feat_names.json"), "w"))
    print("Saved results/clean.csv, X.npy, y.npy, feat_names.json")


if __name__ == "__main__":
    main()
