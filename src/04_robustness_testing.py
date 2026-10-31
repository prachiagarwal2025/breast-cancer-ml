"""
04 - Robustness stress tests: Gaussian noise injection, feature dropout,
bootstrap CI, calibration, and a learning curve.

Note on noise injection (Section 6.1 of the report): this is a robustness
EXPERIMENT, not a synthetic-data generation technique. The noised copies of
the test set are used once, in memory, to see how sensitive predictions are
to small perturbations - they are never treated as new "real" data, never
added to the training set, and never used to inflate the reported dataset
size. Framing it as "generating synthetic data" would overstate what this is.

Every model here is trained inside a Pipeline with a SimpleImputer step, so the
median used to fill missing `bare_nuclei` values is fit on the training set
only, same as in 02_model_training.py.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from sklearn.model_selection import learning_curve, StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(FIG_DIR, exist_ok=True)
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

sns.set_theme(style="whitegrid", palette="Set2")

best_pipe = joblib.load(f"{OUT_DIR}/best_model.joblib")
with open(f"{OUT_DIR}/best_model_name.txt") as f:
    best_name = f.read().strip()

X_train = pd.read_csv(f"{OUT_DIR}/X_train.csv")
X_test = pd.read_csv(f"{OUT_DIR}/X_test.csv")
y_train = pd.read_csv(f"{OUT_DIR}/y_train.csv").squeeze()
y_test = pd.read_csv(f"{OUT_DIR}/y_test.csv").squeeze()


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
fitted = {}
for name, clf in models.items():
    pipe = make_pipeline(clf)
    pipe.fit(X_train, y_train)
    fitted[name] = pipe

# --- 6.1 Gaussian noise injection (robustness probe, not synthetic data) ---
noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
noise_results = {name: [] for name in fitted}

feat_std = X_test.std()
for sigma_frac in noise_levels:
    noisy_X = X_test + rng.normal(0, sigma_frac * feat_std.values, size=X_test.shape)
    for name, pipe in fitted.items():
        proba = pipe.predict_proba(noisy_X)[:, 1]
        auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, pipe.predict(noisy_X))
        noise_results[name].append({"sigma_frac": sigma_frac, "auc": auc, "accuracy": acc})

noise_df = pd.concat({name: pd.DataFrame(v) for name, v in noise_results.items()}, names=["model"]).reset_index(level=0)
noise_df.to_csv(f"{OUT_DIR}/robustness_noise_injection.csv", index=False)

plt.figure(figsize=(8, 5.5))
for name in fitted:
    sub = noise_df[noise_df.model == name]
    plt.plot(sub.sigma_frac, sub.auc, marker="o", label=name)
plt.xlabel("Injected Gaussian noise (fraction of feature std-dev)")
plt.ylabel("ROC-AUC on noised test set")
plt.title("Robustness Experiment: Sensitivity to Input Noise\n(in-memory perturbation of the test set, not synthetic training data)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/robustness_noise_injection.png", dpi=150)
plt.close()

# --- 6.2 Feature dropout / occlusion ---------------------------------------
importance = pd.read_csv(f"{OUT_DIR}/shap_global_importance.csv", index_col=0)
ranked_feats = importance.index.tolist()

dropout_ks = [0, 1, 2, 3, 5, 7]
dropout_results = []
for k in dropout_ks:
    X_occluded = X_test.copy()
    feats_to_drop = ranked_feats[:k]
    for feat in feats_to_drop:
        X_occluded[feat] = X_train[feat].mean()
    for name, pipe in fitted.items():
        proba = pipe.predict_proba(X_occluded)[:, 1]
        auc = roc_auc_score(y_test, proba)
        dropout_results.append({"model": name, "k_features_dropped": k, "auc": auc})

dropout_df = pd.DataFrame(dropout_results)
dropout_df.to_csv(f"{OUT_DIR}/robustness_feature_dropout.csv", index=False)

plt.figure(figsize=(8, 5.5))
for name in fitted:
    sub = dropout_df[dropout_df.model == name]
    plt.plot(sub.k_features_dropped, sub.auc, marker="o", label=name)
plt.xlabel("Number of top-important features masked (mean-imputed)")
plt.ylabel("ROC-AUC")
plt.title("Robustness to Missing / Occluded Features")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/robustness_feature_dropout.png", dpi=150)
plt.close()

# --- 6.3 Bootstrap stability -------------------------------------------------
n_boot = 500
boot_results = {name: [] for name in fitted}
idx_all = np.arange(len(X_test))
for b in range(n_boot):
    boot_idx = resample(idx_all, replace=True, n_samples=len(idx_all), random_state=b)
    Xb, yb = X_test.iloc[boot_idx], y_test.iloc[boot_idx]
    if yb.nunique() < 2:
        continue
    for name, pipe in fitted.items():
        proba = pipe.predict_proba(Xb)[:, 1]
        boot_results[name].append(roc_auc_score(yb, proba))

boot_summary = pd.DataFrame({
    name: {"mean_auc": np.mean(v), "std_auc": np.std(v),
           "ci_lower_2.5%": np.percentile(v, 2.5), "ci_upper_97.5%": np.percentile(v, 97.5)}
    for name, v in boot_results.items()
}).T
boot_summary.to_csv(f"{OUT_DIR}/robustness_bootstrap_ci.csv")
print("=== Bootstrap 95% CI for ROC-AUC (test set, 500 resamples) ===")
print(boot_summary)

plt.figure(figsize=(8, 5.5))
plot_data = pd.concat({name: pd.Series(v) for name, v in boot_results.items()}, names=["model"]).reset_index(level=0, name="auc")
sns.boxplot(data=plot_data, x="model", y="auc")
plt.xticks(rotation=20, ha="right")
plt.title("Bootstrap Distribution of Test ROC-AUC (500 resamples)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/robustness_bootstrap_boxplot.png", dpi=150)
plt.close()

# --- 6.4 Calibration ----------------------------------------------------------
proba_best = best_pipe.predict_proba(X_test)[:, 1]
brier = brier_score_loss(y_test, proba_best)
frac_pos, mean_pred = calibration_curve(y_test, proba_best, n_bins=10, strategy="quantile")

plt.figure(figsize=(6, 6))
plt.plot(mean_pred, frac_pos, marker="o", label=f"{best_name} (Brier={brier:.4f})")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed frequency of malignancy")
plt.title("Calibration / Reliability Diagram")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/calibration_curve.png", dpi=150)
plt.close()

# --- 6.5 Learning curve --------------------------------------------------------
train_sizes, train_scores, val_scores = learning_curve(
    best_pipe, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
    scoring="roc_auc", train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1
)
plt.figure(figsize=(7.5, 5.5))
plt.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Training score")
plt.plot(train_sizes, val_scores.mean(axis=1), marker="o", label="Validation score")
plt.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                  val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.2)
plt.xlabel("Training set size")
plt.ylabel("ROC-AUC")
plt.title(f"Learning Curve - {best_name}")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/learning_curve.png", dpi=150)
plt.close()

summary = {
    "best_model": best_name,
    "brier_score": brier,
    "auc_at_0pct_noise": float(noise_df[(noise_df.model == best_name) & (noise_df.sigma_frac == 0.0)].auc.iloc[0]),
    "auc_at_20pct_noise": float(noise_df[(noise_df.model == best_name) & (noise_df.sigma_frac == 0.20)].auc.iloc[0]),
    "auc_drop_at_20pct_noise": float(
        noise_df[(noise_df.model == best_name) & (noise_df.sigma_frac == 0.0)].auc.iloc[0] -
        noise_df[(noise_df.model == best_name) & (noise_df.sigma_frac == 0.20)].auc.iloc[0]
    ),
    "bootstrap_ci_95": boot_summary.loc[best_name].to_dict(),
}
with open(f"{OUT_DIR}/robustness_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nRobustness testing complete.")
print(json.dumps(summary, indent=2))
