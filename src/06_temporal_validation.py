"""
06 - Chronological holdout / distribution-shift analysis.

This is NOT external validation: both cohorts below still come from the same
institution, the same data collection process, and the same patient
population, just different points in time. What this DOES provide is an
out-of-time check (does performance hold up on genuinely later, unseen
patients) and a look at how the model behaves under a real covariate/label
shift between the two cohorts. See the README for why this distinction
matters.

Known limitations (quantified in 07_validation_audit.py):
  - `best_name` comes from 02_model_training.py's CV over a random 80% split that
    includes later-cohort patients, so model-family choice has seen "late" data.
    Re-selecting on the early cohort alone picks the same model, so it is harmless
    here, but it is not a strictly clean out-of-time selection.
  - A single split with no interval; no degradation observed is not proof of robustness.

Implementation notes:
  - Explicitly re-sorts by `id` before splitting, instead of trusting that the
    cleaned CSV's row order is chronological. 01_data_eda.py already saves the
    file in id-sorted order, but sorting again here makes this script correct
    on its own, even if someone re-generates the cleaned CSV differently later.
  - Fits the SimpleImputer used for missing `bare_nuclei` values on the early
    (development) cohort only, then applies it to the late (holdout) cohort -
    consistent with how imputation is handled everywhere else in this project.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
OUT_DIR = os.path.join(ROOT, "outputs")
DATA_DIR = os.path.join(ROOT, "data")
RANDOM_STATE = 42

sns.set_theme(style="whitegrid", palette="Set2")

df = pd.read_csv(f"{DATA_DIR}/wisconsin_original_clean.csv")
# Defensive re-sort: id is our chronology proxy (see 01_data_eda.py for why
# raw row order alone isn't trustworthy). This script does not assume the
# CSV on disk is already sorted.
df = df.sort_values("id", kind="stable").reset_index(drop=True)
feature_cols = [c for c in df.columns if c not in ("id", "diagnosis")]

with open(f"{OUT_DIR}/best_model_name.txt") as f:
    best_name = f.read().strip()

model_lookup = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
}

split_point = int(len(df) * 0.7)
early = df.iloc[:split_point].reset_index(drop=True)
late = df.iloc[split_point:].reset_index(drop=True)

print(f"Early (development) cohort: {len(early)} patients, malignant rate = {(early.diagnosis == 'malignant').mean():.3f}")
print(f"Late (temporal holdout) cohort: {len(late)} patients, malignant rate = {(late.diagnosis == 'malignant').mean():.3f}")

X_early, y_early = early[feature_cols], (early["diagnosis"] == "malignant").astype(int)
X_late, y_late = late[feature_cols], (late["diagnosis"] == "malignant").astype(int)

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # fit on X_early only via .fit() below
    ("scaler", StandardScaler()),
    ("clf", model_lookup[best_name]),
])
pipe.fit(X_early, y_early)

y_pred_late = pipe.predict(X_late)
y_proba_late = pipe.predict_proba(X_late)[:, 1]

temporal_metrics = {
    "accuracy": accuracy_score(y_late, y_pred_late),
    "precision": precision_score(y_late, y_pred_late),
    "recall": recall_score(y_late, y_pred_late),
    "f1": f1_score(y_late, y_pred_late),
    "roc_auc": roc_auc_score(y_late, y_proba_late),
}
print("\n=== Chronological holdout (distribution-shift) performance ===")
for k, v in temporal_metrics.items():
    print(f"{k}: {v:.4f}")

main_metrics = json.load(open(f"{OUT_DIR}/test_metrics.json"))
comparison = {
    "random_split_test": {k: main_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]},
    "temporal_holdout_test": temporal_metrics,
    "auc_gap": main_metrics["roc_auc"] - temporal_metrics["roc_auc"],
}
with open(f"{OUT_DIR}/temporal_validation_summary.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "early_n": len(early),
        "late_n": len(late),
        "early_malignant_rate": float((early.diagnosis == "malignant").mean()),
        "late_malignant_rate": float((late.diagnosis == "malignant").mean()),
        **comparison,
    }, f, indent=2)

cm = confusion_matrix(y_late, y_pred_late)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["benign", "malignant"])
fig, ax = plt.subplots(figsize=(6, 5.5))
disp.plot(ax=ax, cmap="Oranges", colorbar=False)
ax.set_title(f"Confusion Matrix - Temporal Holdout ({best_name})", fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/temporal_confusion_matrix.png", dpi=150)
plt.close()

compare_df = pd.DataFrame(comparison["random_split_test"], index=["Random-split test"]).T
compare_df["Temporal holdout"] = pd.Series(comparison["temporal_holdout_test"])
compare_df = compare_df.rename(columns={0: "Random-split test"})

plt.figure(figsize=(7.5, 5.5))
compare_df.plot(kind="bar", ax=plt.gca())
plt.ylim(0.7, 1.02)
plt.ylabel("Score")
plt.title(f"Random-Split Test vs. Temporal Holdout ({best_name})")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/temporal_vs_random_comparison.png", dpi=150)
plt.close()

print("\nTemporal validation complete. Figures and summary saved.")
print(json.dumps(comparison, indent=2))
