"""
03 - SHAP global + local explanations for the best model.

Design notes:
  - SHAP is computed on the model's actual input space (imputed + scaled,
    since that's what the classifier was fit on), but plots are LABELED and
    COLORED using imputed-but-original-unit feature values (e.g. "clump
    thickness = 7", not "clump thickness = 1.42 std-devs"). Standardized
    z-scores on a 1-10 clinical grading scale were not interpretable to a
    reader who isn't also tracking each feature's mean/std.
  - The set of rows actually explained is fixed once (`N_EXPLAIN`) and used
    consistently everywhere in this script, so the waterfall plots always draw
    from the same rows as the summary plots.
  - Tree/linear models (fast, exact explainers) explain the full test set.
    Only the slower KernelExplainer path (used for SVM / KNN) is capped, and
    that cap is stated explicitly in the printed output and the report.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
OUT_DIR = os.path.join(ROOT, "outputs")
RANDOM_STATE = 42
KERNEL_EXPLAINER_CAP = 150  # only applies to the slow, sampling-based path

best_pipe = joblib.load(f"{OUT_DIR}/best_model.joblib")
X_train = pd.read_csv(f"{OUT_DIR}/X_train.csv")
X_test = pd.read_csv(f"{OUT_DIR}/X_test.csv")
y_test = pd.read_csv(f"{OUT_DIR}/y_test.csv").squeeze()

imputer = best_pipe.named_steps["imputer"]
scaler = best_pipe.named_steps["scaler"]
clf = best_pipe.named_steps["clf"]
feature_names = X_train.columns.tolist()

# "Display" data: imputed so there are no NaNs to break plotting, but left in
# original clinical units (1-10 grading scale) so SHAP plots stay readable.
X_train_display = pd.DataFrame(imputer.transform(X_train), columns=feature_names)
X_test_display = pd.DataFrame(imputer.transform(X_test), columns=feature_names)

# "Model input" data: what the classifier actually sees (imputed + scaled).
X_train_model = pd.DataFrame(scaler.transform(X_train_display.values), columns=feature_names)
X_test_model = pd.DataFrame(scaler.transform(X_test_display.values), columns=feature_names)

model_type = type(clf).__name__
print(f"Explaining model type: {model_type}")

n_explain = len(X_test_model)
explain_note = "full held-out test set"

if model_type in ("RandomForestClassifier", "GradientBoostingClassifier"):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_model)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1]
    X_explained_model = X_test_model
    X_explained_display = X_test_display
elif model_type == "LogisticRegression":
    background = shap.sample(X_train_model, 100, random_state=RANDOM_STATE)
    explainer = shap.LinearExplainer(clf, background)
    shap_values = explainer.shap_values(X_test_model)
    base_value = explainer.expected_value
    X_explained_model = X_test_model
    X_explained_display = X_test_display
else:
    # SVM / KNN: no closed-form attribution, fall back to sampling-based
    # KernelExplainer, which is expensive - cap the explained rows and say so.
    n_explain = min(KERNEL_EXPLAINER_CAP, len(X_test_model))
    explain_note = f"first {n_explain} of {len(X_test_model)} test rows (KernelExplainer is sampling-based and slow)"
    background = shap.sample(X_train_model, 100, random_state=RANDOM_STATE)
    explainer = shap.KernelExplainer(clf.predict_proba, background)
    X_explained_model = X_test_model.iloc[:n_explain].reset_index(drop=True)
    X_explained_display = X_test_display.iloc[:n_explain].reset_index(drop=True)
    sv = explainer.shap_values(X_explained_model, nsamples=200)
    shap_values = sv[1] if isinstance(sv, list) else sv
    base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

print(f"Rows explained: {explain_note}")

shap_values = np.array(shap_values)
if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]

y_true_slice = y_test.iloc[:n_explain].reset_index(drop=True)
y_pred_slice = pd.Series(best_pipe.predict(X_test).astype(int)).iloc[:n_explain].reset_index(drop=True)

plt.figure()
shap.summary_plot(shap_values, X_explained_display, show=False, max_display=15)
plt.title(f"SHAP Summary (Beeswarm) - {model_type}\n(feature values in original clinical units, 1-10 scale)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure()
shap.summary_plot(shap_values, X_explained_display, plot_type="bar", show=False, max_display=15)
plt.title(f"Mean |SHAP value| - Global Feature Importance ({model_type})")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/shap_summary_bar.png", dpi=150, bbox_inches="tight")
plt.close()

mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names)
mean_abs_shap.sort_values(ascending=False).to_csv(f"{OUT_DIR}/shap_global_importance.csv",
                                                     header=["mean_abs_shap_value"])
print("\nFeatures ranked by mean |SHAP|:")
print(mean_abs_shap.sort_values(ascending=False))

correct_idx = (y_true_slice == y_pred_slice)
wrong_idx = ~correct_idx


def plot_waterfall(idx, tag):
    exp = shap.Explanation(values=shap_values[idx], base_values=base_value,
                            data=X_explained_display.iloc[idx].values,
                            feature_names=feature_names)
    plt.figure()
    shap.plots.waterfall(exp, show=False, max_display=12)
    plt.title(f"Local Explanation - {tag} (true={y_true_slice.iloc[idx]}, pred={y_pred_slice.iloc[idx]})")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/shap_waterfall_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()


if correct_idx.any():
    plot_waterfall(int(np.where(correct_idx)[0][0]), "correct_case")
if wrong_idx.any():
    plot_waterfall(int(np.where(wrong_idx)[0][0]), "misclassified_case")
else:
    print("No misclassified instances in the explained subset - skipping that plot.")

with open(f"{OUT_DIR}/shap_explain_note.txt", "w") as f:
    f.write(explain_note + "\n")

print("\nExplainability analysis complete. Figures saved to", FIG_DIR)
