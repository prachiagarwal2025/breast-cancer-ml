"""
02 - Train & compare five classifiers, pick the best by CV ROC-AUC, evaluate once
on a held-out test set.

Median imputation for `bare_nuclei` lives INSIDE the sklearn Pipeline, as a
SimpleImputer step before StandardScaler and the classifier. That means:
  - during 10-fold CV, the median is recomputed on each training fold and
    applied to that fold's validation fold only;
  - the final median used at test time is fit on the full training set only,
    never on the test set.
Computing the median once on the whole dataset before the split would let
test-set values influence the numbers used to fill training-set rows.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.utils import resample
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
MODEL_DIR = os.path.join(ROOT, "outputs")
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(FIG_DIR, exist_ok=True)
RANDOM_STATE = 42
N_BOOT = 2000

sns.set_theme(style="whitegrid", palette="Set2")

df = pd.read_csv(f"{DATA_DIR}/wisconsin_original_clean.csv")
feature_cols = [c for c in df.columns if c not in ("id", "diagnosis")]
X = df[feature_cols]
y = (df["diagnosis"] == "malignant").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)


def make_pipeline(clf):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
}

scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

cv_results = {}
for name, clf in models.items():
    pipe = make_pipeline(clf)
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    cv_results[name] = {m: (scores[f"test_{m}"].mean(), scores[f"test_{m}"].std()) for m in scoring}

cv_summary = pd.DataFrame({name: {m: f"{v[0]:.4f} +/- {v[1]:.4f}" for m, v in res.items()}
                            for name, res in cv_results.items()}).T
cv_summary.to_csv(f"{MODEL_DIR}/cv_model_comparison.csv")
print("=== 10-fold Stratified CV results (train set, imputer fit per-fold) ===")
print(cv_summary)

plot_df = pd.DataFrame({
    name: {m: cv_results[name][m][0] for m in scoring} for name in models
}).T.reset_index().rename(columns={"index": "model"})
plot_melt = plot_df.melt(id_vars="model", var_name="metric", value_name="score")

plt.figure(figsize=(11, 6))
sns.barplot(data=plot_melt, x="model", y="score", hue="metric")
plt.xticks(rotation=20, ha="right")
plt.ylim(0.85, 1.0)
plt.title("10-Fold Cross-Validation: Model Comparison")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/cv_model_comparison.png", dpi=150)
plt.close()

best_name = max(cv_results, key=lambda n: cv_results[n]["roc_auc"][0])
print(f"\nBest model by CV ROC-AUC: {best_name}")

best_pipe = make_pipeline(models[best_name])
best_pipe.fit(X_train, y_train)

y_pred = best_pipe.predict(X_test)
y_proba = best_pipe.predict_proba(X_test)[:, 1]

test_metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
}
print("\n=== Held-out test set performance (best model, default 0.5 threshold) ===")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")

# --- 95% confidence intervals via bootstrap resampling of the test set -----
# Answers "how much would this number move if we'd happened to draw a
# different sample of test patients", NOT "how confident is the model".
X_test_arr = X_test.reset_index(drop=True)
y_test_arr = y_test.reset_index(drop=True)
rng_idx = np.arange(len(X_test_arr))
boot_metric_samples = {m: [] for m in scoring}
for b in range(N_BOOT):
    boot_idx = resample(rng_idx, replace=True, n_samples=len(rng_idx), random_state=b)
    yb = y_test_arr.iloc[boot_idx]
    if yb.nunique() < 2:
        continue
    Xb = X_test_arr.iloc[boot_idx]
    proba_b = best_pipe.predict_proba(Xb)[:, 1]
    pred_b = (proba_b >= 0.5).astype(int)
    boot_metric_samples["accuracy"].append(accuracy_score(yb, pred_b))
    boot_metric_samples["precision"].append(precision_score(yb, pred_b, zero_division=0))
    boot_metric_samples["recall"].append(recall_score(yb, pred_b, zero_division=0))
    boot_metric_samples["f1"].append(f1_score(yb, pred_b, zero_division=0))
    boot_metric_samples["roc_auc"].append(roc_auc_score(yb, proba_b))

test_metrics_ci = {
    m: {
        "point_estimate": test_metrics[m],
        "ci_lower_2.5%": float(np.percentile(boot_metric_samples[m], 2.5)),
        "ci_upper_97.5%": float(np.percentile(boot_metric_samples[m], 97.5)),
    }
    for m in scoring
}
print(f"\n=== 95% bootstrap CIs for held-out test metrics (n={N_BOOT} resamples) ===")
for m, v in test_metrics_ci.items():
    print(f"{m}: {v['point_estimate']:.4f}  [{v['ci_lower_2.5%']:.4f}, {v['ci_upper_97.5%']:.4f}]")

with open(f"{MODEL_DIR}/test_metrics.json", "w") as f:
    json.dump({
        "best_model": best_name,
        **test_metrics,
        "bootstrap_ci_95": test_metrics_ci,
        "bootstrap_n_resamples": N_BOOT,
    }, f, indent=2)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["benign", "malignant"])
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix - {best_name} (Test Set)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/confusion_matrix.png", dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"{best_name} (AUC = {test_metrics['roc_auc']:.3f})", color="crimson")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Best Model on Held-out Test Set")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/roc_curve.png", dpi=150)
plt.close()

joblib.dump(best_pipe, f"{MODEL_DIR}/best_model.joblib")
X_train.to_csv(f"{MODEL_DIR}/X_train.csv", index=False)
X_test.to_csv(f"{MODEL_DIR}/X_test.csv", index=False)
y_train.to_csv(f"{MODEL_DIR}/y_train.csv", index=False)
y_test.to_csv(f"{MODEL_DIR}/y_test.csv", index=False)

with open(f"{MODEL_DIR}/best_model_name.txt", "w") as f:
    f.write(best_name)

print("\nModel training complete. Best model & data splits saved to", MODEL_DIR)
