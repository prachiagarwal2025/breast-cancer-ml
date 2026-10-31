"""
07 - Validation audit: stress-tests the headline claims made by scripts 02, 05, 06.

Nothing here changes the model, the split, or the committed threshold. It asks:

  A. Is the 99.7% ROC-AUC inflated by exact-duplicate feature vectors that sit on both
     sides of the random split?  (The dataset has only 442 distinct feature vectors
     across 645 patients.)
  B. How much does the temporal-holdout conclusion depend on (i) using `id` order vs.
     raw file order as the clock, (ii) the cut point, and (iii) the model family having
     been picked by CV that included the "later" patients?
  C. How fragile is the zero-false-negative threshold, and what does it cost in false
     positives / PPV at realistic prevalences?
  D. How separable is the task for trivial baselines?  (context for the AUC)

Results -> outputs/validation_audit_summary.json
"""
import json
import os
import numpy as np
import pandas as pd
import joblib
from scipy.stats import beta
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
DATA = os.path.join(ROOT, "data")
SEED = 42
rng = np.random.default_rng(SEED)


def pipe(clf):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("clf", clf)])


def lr():
    return pipe(LogisticRegression(max_iter=5000, random_state=SEED))


def boot_auc_ci(y, p, n=2000):
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        vals.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def cp_interval(k, n, alpha=0.05):
    """Clopper-Pearson exact binomial interval."""
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return [float(lo), float(hi)]


def key(X):
    return X.fillna(-1).astype(int).astype(str).agg("-".join, axis=1)


summary = {}

# ---------------------------------------------------------------- data
df = pd.read_csv(f"{DATA}/wisconsin_original_clean.csv").sort_values("id", kind="stable").reset_index(drop=True)
feats = [c for c in df.columns if c not in ("id", "diagnosis")]
df["y"] = (df.diagnosis == "malignant").astype(int)

# ================================================================ A. duplicates
vec = key(df[feats])
summary["A_duplicates"] = {
    "patients": int(len(df)),
    "distinct_feature_vectors": int(vec.nunique()),
    "patients_sharing_a_vector_with_another_patient": int((vec.duplicated(keep=False)).sum()),
    "label_conflicting_vectors": int((df.groupby(vec)["y"].nunique() > 1).sum()),
}

Xtr = pd.read_csv(f"{OUT}/X_train.csv"); ytr = pd.read_csv(f"{OUT}/y_train.csv").squeeze()
Xte = pd.read_csv(f"{OUT}/X_test.csv");  yte = pd.read_csv(f"{OUT}/y_test.csv").squeeze()
model = joblib.load(f"{OUT}/best_model.joblib")
p_te = model.predict_proba(Xte)[:, 1]
twin = key(Xte).isin(set(key(Xtr))).values
novel = ~twin
a = {"test_n": int(len(yte)),
     "test_patients_with_exact_twin_in_train": int(twin.sum()),
     "test_patients_novel": int(novel.sum()),
     "auc_all": float(roc_auc_score(yte, p_te))}
for name, m in [("twin", twin), ("novel", novel)]:
    y_m, p_m = yte.values[m], p_te[m]
    d = {"n": int(m.sum()), "n_malignant": int(y_m.sum()),
         "errors_at_0.5": int(((p_m >= .5).astype(int) != y_m).sum())}
    if len(set(y_m)) == 2:
        d["auc"] = float(roc_auc_score(y_m, p_m)); d["auc_ci95"] = boot_auc_ci(y_m, p_m)
    a[name] = d
summary["A_duplicates"]["random_split_test_by_twin_status"] = a

# Group-aware split: keep every copy of a feature vector on one side of the split.
from sklearn.model_selection import StratifiedGroupKFold
grp = pd.factorize(vec)[0]
oof_rand = cross_val_predict(lr(), df[feats], df.y,
                             cv=StratifiedKFold(10, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
oof_grp = np.zeros(len(df))
for tr, te in StratifiedGroupKFold(10, shuffle=True, random_state=SEED).split(df[feats], df.y, grp):
    oof_grp[te] = clone(lr()).fit(df.loc[tr, feats], df.y[tr]).predict_proba(df.loc[te, feats])[:, 1]
summary["A_duplicates"]["oof_auc_random_10fold_all_data"] = float(roc_auc_score(df.y, oof_rand))
summary["A_duplicates"]["oof_auc_group_10fold_all_data"] = float(roc_auc_score(df.y, oof_grp))

# ================================================================ B. temporal
B = {}
# (i) id order vs raw file order as the clock
raw = pd.read_csv(f"{DATA}/wisconsin_original_raw.data", header=None,
                  names=["id"] + feats + ["cls"])
raw["bare_nuclei"] = pd.to_numeric(raw["bare_nuclei"].replace("?", np.nan))
raw["y"] = (raw.cls == 4).astype(int)
file_order = raw.drop_duplicates("id", keep="first").reset_index(drop=True)   # file order kept
clocks = {"id_order": df, "raw_file_order": file_order}


def temporal(frame, frac):
    k = int(len(frame) * frac)
    tr, te = frame.iloc[:k], frame.iloc[k:]
    m = clone(lr()).fit(tr[feats], tr.y)
    p = m.predict_proba(te[feats])[:, 1]
    sh = key(te[feats]).isin(set(key(tr[feats]))).values
    return {"train_n": int(k), "test_n": int(len(te)), "test_malignant_rate": float(te.y.mean()),
            "auc": float(roc_auc_score(te.y, p)), "auc_ci95": boot_auc_ci(te.y, p),
            "test_share_with_exact_twin_in_train": float(sh.mean()),
            "auc_on_novel_vectors_only": float(roc_auc_score(te.y[~sh], p[~sh])) if len(set(te.y[~sh])) == 2 else None,
            "n_novel": int((~sh).sum())}


B["single_cut_by_clock_and_fraction"] = {
    clock: {f"train_{int(f*100)}pct": temporal(fr, f) for f in (0.5, 0.6, 0.7, 0.8)}
    for clock, fr in clocks.items()}

# rolling-origin: expanding window, next-100-patients blocks (LR pre-specified)
roll = []
for k in range(245, 545 + 1, 100):
    tr, te = df.iloc[:k], df.iloc[k:k + 100]
    p = clone(lr()).fit(tr[feats], tr.y).predict_proba(te[feats])[:, 1]
    roll.append({"train_n": k, "test_n": int(len(te)), "test_malignant": int(te.y.sum()),
                 "auc": float(roc_auc_score(te.y, p))})
B["rolling_origin_id_order"] = roll

# (iii) model-family choice made on the EARLY cohort only
k = int(len(df) * .7); early = df.iloc[:k]
cands = {"Logistic Regression": lr(),
         "SVM (RBF)": pipe(SVC(probability=True, random_state=SEED)),
         "Random Forest": pipe(RandomForestClassifier(300, random_state=SEED)),
         "Gradient Boosting": pipe(GradientBoostingClassifier(random_state=SEED)),
         "K-Nearest Neighbors": pipe(KNeighborsClassifier(7))}
cv = StratifiedKFold(10, shuffle=True, random_state=SEED)
early_cv = {n: float(cross_val_score(m, early[feats], early.y, cv=cv, scoring="roc_auc").mean())
            for n, m in cands.items()}
B["model_selection_on_early_cohort_only"] = {"cv_auc": early_cv, "winner": max(early_cv, key=early_cv.get)}

# committed-threshold procedure, run temporally (derive on early OOF, apply once to late)
late = df.iloc[k:]
oof_e = cross_val_predict(lr(), early[feats], early.y, cv=cv, method="predict_proba")[:, 1]
t_early = float(oof_e[early.y.values == 1].min())          # == highest zero-FN threshold on OOF
p_late = clone(lr()).fit(early[feats], early.y).predict_proba(late[feats])[:, 1]
tn, fp, fn, tp = confusion_matrix(late.y, (p_late >= t_early).astype(int)).ravel()
tn5, fp5, fn5, tp5 = confusion_matrix(late.y, (p_late >= .5).astype(int)).ravel()
B["threshold_procedure_applied_temporally"] = {
    "threshold_from_early_oof": t_early,
    "late_at_that_threshold": {"tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)},
    "late_at_0.5": {"tp": int(tp5), "fn": int(fn5), "fp": int(fp5), "tn": int(tn5)}}
summary["B_temporal"] = B

# ================================================================ C. threshold
C = {}
ts = []
for s in range(50):                                          # same procedure, 50 different CV shuffles
    o = cross_val_predict(lr(), Xtr, ytr, cv=StratifiedKFold(10, shuffle=True, random_state=s),
                          method="predict_proba")[:, 1]
    ts.append(float(o[ytr.values == 1].min()))
C["threshold_across_50_cv_shuffles"] = {"min": min(ts), "median": float(np.median(ts)), "max": max(ts)}
oof = cross_val_predict(lr(), Xtr, ytr, cv=StratifiedKFold(10, shuffle=True, random_state=SEED),
                        method="predict_proba")[:, 1]
t = float(oof[ytr.values == 1].min())
C["threshold_is_set_by_one_training_patient"] = {
    "committed_threshold": t,
    "train_benign_flagged_at_threshold": int(((oof >= t) & (ytr.values == 0)).sum()),
    "train_benign_total": int((ytr.values == 0).sum())}

rows = []
for name, thr in (("0.5", .5), (f"{t:.4f}", t)):
    tn, fp, fn, tp = confusion_matrix(yte, (p_te >= thr).astype(int)).ravel()
    sens_ci, spec_ci = cp_interval(int(tp), int(tp + fn)), cp_interval(int(tn), int(tn + fp))
    rows.append({"threshold": name, "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
                 "sensitivity": tp / (tp + fn), "sensitivity_ci95_exact": sens_ci,
                 "specificity": tn / (tn + fp), "specificity_ci95_exact": spec_ci,
                 "ppv_in_this_test_set": tp / (tp + fp) if tp + fp else None})
C["test_confusion_by_threshold"] = rows
spec_lo = rows[1]["specificity"]
C["ppv_at_other_prevalences_using_committed_threshold"] = {
    f"prevalence_{int(pv*100)}pct": float(pv * 1.0 / (pv * 1.0 + (1 - pv) * (1 - spec_lo)))
    for pv in (0.01, 0.05, 0.10, 0.36)}
C["ppv_note"] = ("assumes sensitivity=1.0 and specificity as measured on this test set; both are uncertain "
                 "(see exact CIs). Illustrates why a precision measured at 36% prevalence does not transfer.")
summary["C_threshold"] = C

# ================================================================ D. separability
D = {}
for f in feats:
    s = df[f].fillna(df[f].median())
    D[f] = float(roc_auc_score(df.y, s))
D["_sum_of_all_nine_features_no_model"] = float(roc_auc_score(df.y, df[feats].fillna(df[feats].median()).sum(axis=1)))
summary["D_single_feature_auc_all_data_no_fitting"] = D

with open(f"{OUT}/validation_audit_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
