"""
05 - Decision threshold tuning.

The threshold is chosen without looking at the held-out test set. Picking the
candidate thresholds, the precision/recall curve or the "zero false negative"
threshold from test-set predictions would tune a modeling decision on the test
set, and the final "test" numbers would stop being an estimate of performance
on unseen data.

Approach:
  1. Generate out-of-fold (OOF) predicted probabilities on the TRAINING set
     only, via cross_val_predict with the same CV scheme used for model
     selection. Every training patient gets a probability from a model that
     never saw that patient during fitting - so this is still an
     unseen-data estimate, but it never touches the test set.
  2. Do all threshold exploration (the precision/recall curve, the candidate
     threshold table, the "zero false negative" search) on these OOF
     probabilities only.
  3. Take the single threshold this analysis recommends, refit the pipeline
     on the full training set, and apply that ALREADY-CHOSEN threshold to the
     test set exactly once, purely to report how it performs. No threshold
     search happens on test-set predictions at any point.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
OUT_DIR = os.path.join(ROOT, "outputs")
RANDOM_STATE = 42

sns.set_theme(style="whitegrid", palette="Set2")

best_pipe = joblib.load(f"{OUT_DIR}/best_model.joblib")
with open(f"{OUT_DIR}/best_model_name.txt") as f:
    best_name = f.read().strip()

X_train = pd.read_csv(f"{OUT_DIR}/X_train.csv")
y_train = pd.read_csv(f"{OUT_DIR}/y_train.csv").squeeze()
X_test = pd.read_csv(f"{OUT_DIR}/X_test.csv")
y_test = pd.read_csv(f"{OUT_DIR}/y_test.csv").squeeze()

# --- Step 1: OOF probabilities on the training set only ---------------------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
oof_proba = cross_val_predict(best_pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

# --- Step 2: all threshold exploration happens on OOF (train) predictions --
precisions, recalls, thresholds = precision_recall_curve(y_train, oof_proba)

plt.figure(figsize=(7, 6))
plt.plot(recalls, precisions, color="crimson")
plt.scatter(
    recall_score(y_train, oof_proba >= 0.5), precision_score(y_train, oof_proba >= 0.5),
    color="black", zorder=5, label="Default threshold (0.5)"
)
plt.xlabel("Recall (Sensitivity to malignant cases)")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Tradeoff - {best_name}\n(tuned on training-set CV out-of-fold predictions, not the test set)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/precision_recall_curve.png", dpi=150)
plt.close()

candidate_thresholds = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
rows = []
for t in candidate_thresholds:
    pred = (oof_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_train, pred).ravel()
    rows.append({
        "threshold": t,
        "precision": precision_score(y_train, pred, zero_division=0),
        "recall": recall_score(y_train, pred, zero_division=0),
        "f1": f1_score(y_train, pred, zero_division=0),
        "false_negatives": int(fn),
        "false_positives": int(fp),
    })
threshold_df = pd.DataFrame(rows)
threshold_df.to_csv(f"{OUT_DIR}/threshold_tuning.csv", index=False)
print("=== Threshold tuning table (training-set OOF predictions) ===")
print(threshold_df.to_string(index=False))

all_t = sorted(np.unique(np.concatenate([thresholds, [0.0, 1.0]])))
fn_free_candidates = []
for t in all_t:
    pred = (oof_proba >= t).astype(int)
    fn = confusion_matrix(y_train, pred).ravel()[2]
    if fn == 0:
        fn_free_candidates.append(t)
recommended_threshold = max(fn_free_candidates) if fn_free_candidates else 0.5
print(f"\nHighest threshold with zero false negatives on training OOF predictions "
      f"(this is the threshold we commit to BEFORE looking at the test set): {recommended_threshold}")

# --- Step 3: apply the already-chosen threshold to the test set, once ------
test_proba = best_pipe.predict_proba(X_test)[:, 1]
default_pred = (test_proba >= 0.5).astype(int)
recommended_pred = (test_proba >= recommended_threshold).astype(int)

summary = {
    "best_model": best_name,
    "tuning_method": "threshold chosen from 10-fold CV out-of-fold predictions on the training set only; "
                     "applied to the test set exactly once below",
    "default_threshold": 0.5,
    "default_threshold_test_precision": float(precision_score(y_test, default_pred, zero_division=0)),
    "default_threshold_test_recall": float(recall_score(y_test, default_pred, zero_division=0)),
    "recommended_threshold": float(recommended_threshold),
    "recommended_threshold_source": "zero-false-negative point on training-set OOF predictions",
    "recommended_threshold_test_precision": float(precision_score(y_test, recommended_pred, zero_division=0)),
    "recommended_threshold_test_recall": float(recall_score(y_test, recommended_pred, zero_division=0)),
    "recommended_threshold_test_false_negatives": int(confusion_matrix(y_test, recommended_pred).ravel()[2]),
    "recommended_threshold_test_false_positives": int(confusion_matrix(y_test, recommended_pred).ravel()[1]),
    "caveat": "This threshold caught all malignant cases in this particular test split (46 cases; exact 95% CI for sensitivity 92.3-100%), but it is set by a single training patient, varies across CV shuffles, and did not give zero misses on the later temporal cohort. It is "
              "specific to this dataset and would need to be re-tuned (via the same train-only procedure) "
              "on new data, not reused as a fixed clinical cutoff.",
}
with open(f"{OUT_DIR}/threshold_analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Test-set performance at the pre-chosen threshold (evaluated once) ===")
print(json.dumps(summary, indent=2))
print("\nThreshold analysis complete.")
