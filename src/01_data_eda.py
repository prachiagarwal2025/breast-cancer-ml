"""
01 - Load, clean, and explore the Breast Cancer Wisconsin (Original) dataset.

Two things this script deliberately does NOT do:

1. It does NOT impute the missing `bare_nuclei` values here. Filling them in
   before the train/test split means the fill value (e.g. a median) is computed
   using information from patients who will later end up in the test set - a
   classic case of data leakage. Imputation now happens *inside* the modeling
   pipeline (see 02_model_training.py), fit on the training fold/split only.

2. It does NOT rely on raw file row order as a stand-in for chronology. The id
   column (the sample accession number) is only mostly increasing - there are
   several places where the raw file's row order and id order disagree - so
   this script explicitly sorts by id (using original row position only as a
   tie-breaker for genuine duplicate ids) before saving the cleaned dataset.
   06_temporal_validation.py depends on this order being a properly-sorted
   chronological proxy, not just "whatever order was in the file".
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2")

COLS = ["id", "clump_thickness", "uniformity_cell_size", "uniformity_cell_shape",
        "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei",
        "bland_chromatin", "normal_nucleoli", "mitoses", "class"]

raw = pd.read_csv(f"{DATA_DIR}/wisconsin_original_raw.data", header=None, names=COLS)
raw["_original_order"] = np.arange(len(raw))
print("Raw shape:", raw.shape)
print("Duplicate sample IDs:", raw["id"].duplicated().sum())
print("Missing (bare_nuclei == '?'):", (raw["bare_nuclei"] == "?").sum())

# --- Chronological ordering -------------------------------------------------
# id is a sample accession number, not a verified timestamp, so this is an
# approximation of chronology, not the real thing (see README limitations).
# It is a *better* approximation than raw row order, though: sort by id, and
# for the ids that appear more than once (repeat biopsies of the same
# patient), keep the copy that appeared first in the raw file, since that's
# our best available proxy for "which one was recorded first."
raw_sorted = raw.sort_values(["id", "_original_order"], kind="stable").reset_index(drop=True)
n_inversions = int((raw["id"].values[1:] < raw["id"].values[:-1]).sum())
print(f"Row-order vs. id-order mismatches in the raw file: {n_inversions} "
      f"(this is why we sort by id explicitly instead of trusting file order)")

df = raw_sorted.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
df = df.drop(columns=["_original_order"])
print("Shape after sorting by id and dropping repeat-biopsy duplicates:", df.shape)

# --- Missing values: flagged here, imputed later, inside the pipeline ------
df["bare_nuclei"] = df["bare_nuclei"].replace("?", np.nan).astype(float)
n_missing = int(df["bare_nuclei"].isna().sum())
print(f"{n_missing} records have a missing bare_nuclei value (left as NaN; "
      f"imputed downstream using only the training fold's median)")

df["diagnosis"] = df["class"].map({2: "benign", 4: "malignant"})
df.drop(columns=["class"], inplace=True)

# Saved in chronological (id-sorted) order - 06_temporal_validation.py relies on this.
df.to_csv(f"{DATA_DIR}/wisconsin_original_clean.csv", index=False)

feature_cols = [c for c in df.columns if c not in ("id", "diagnosis")]

print("\nFinal shape:", df.shape)
print("\nClass balance:\n", df["diagnosis"].value_counts())

plt.figure(figsize=(5, 4))
ax = sns.countplot(data=df, x="diagnosis", order=["benign", "malignant"])
ax.set_title("Class Distribution")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/class_balance.png", dpi=150)
plt.close()

# For EDA plots only (correlation heatmap, distribution plots) we need a
# complete-case-ish numeric frame to visualize. This median fill is used ONLY
# for these illustrative figures and is never written back to the file that
# feeds the modeling scripts, so it cannot leak into any model or metric.
eda_view = df.copy()
eda_view["bare_nuclei"] = eda_view["bare_nuclei"].fillna(eda_view["bare_nuclei"].median())

corr_df = eda_view[feature_cols].copy()
corr_df["target_bin"] = (df["diagnosis"] == "malignant").astype(int)
correlations = corr_df.corr()["target_bin"].drop("target_bin").sort_values(key=abs, ascending=False)

plt.figure(figsize=(8, 6.5))
sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
            cbar_kws={"label": "correlation"})
plt.title("Correlation Heatmap - Cytology Features vs. Malignancy")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/correlation_heatmap.png", dpi=150)
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
top6 = correlations.head(6).index.tolist()
for ax, feat in zip(axes.flatten(), top6):
    sns.kdeplot(data=eda_view, x=feat, hue="diagnosis", fill=True, alpha=0.4, ax=ax, legend=(feat == top6[0]))
    ax.set_title(feat.replace("_", " ").title())
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/feature_distributions.png", dpi=150)
plt.close()

correlations.to_csv(f"{DATA_DIR}/feature_target_correlation.csv", header=["abs_correlation_with_malignant"])

print("\nFeatures ranked by correlation with malignancy (EDA view, median-filled for plotting only):")
print(correlations)
print("\nEDA complete. Figures saved to", FIG_DIR)
