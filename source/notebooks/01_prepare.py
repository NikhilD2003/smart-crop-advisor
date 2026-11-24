# %% [markdown]
# # 01 - Data preparation and analysis
# Cleans both datasets, runs descriptive / exploratory / inferential analysis.
# Run from anywhere: `python source/notebooks/01_prepare.py` (also opens as cells in PyCharm/VS Code).

# %%
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
FIG = ROOT / "source" / "models" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
report = {}

# %% [markdown]
# ## Crop recommendation data: check missing, duplicates, impossible values

# %%
crop = pd.read_csv(RAW / "crop_recommendation.csv")
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
report["crop_raw"] = {
    "rows": len(crop),
    "missing": int(crop.isna().sum().sum()),
    "duplicates": int(crop.duplicated().sum()),
    "ph_out_of_0_14": int((~crop.ph.between(0, 14)).sum()),
    "humidity_out_of_0_100": int((~crop.humidity.between(0, 100)).sum()),
    "negative_values": int((crop[FEATURES] < 0).any(axis=1).sum()),
}
crop = crop.drop_duplicates().dropna()
crop = crop[crop.ph.between(0, 14) & crop.humidity.between(0, 100) & (crop[FEATURES] >= 0).all(axis=1)]
crop.to_csv(OUT / "crop_clean.csv", index=False)
report["crop_clean_rows"] = len(crop)

# %% [markdown]
# ## Descriptive + exploratory

# %%
report["describe"] = crop[FEATURES].describe().round(3).to_dict()
report["samples_per_crop"] = crop.label.value_counts().to_dict()
report["mean_per_crop"] = crop.groupby("label")[FEATURES].mean().round(2).to_dict("index")

sns.heatmap(crop[FEATURES].corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Feature correlation"); plt.tight_layout(); plt.savefig(FIG / "correlation.png", dpi=120); plt.close()

crop[FEATURES].hist(bins=30, figsize=(11, 7))
plt.tight_layout(); plt.savefig(FIG / "histograms.png", dpi=120); plt.close()

plt.figure(figsize=(12, 5))
sns.boxplot(data=crop, x="label", y="rainfall")
plt.xticks(rotation=70); plt.tight_layout(); plt.savefig(FIG / "rainfall_by_crop.png", dpi=120); plt.close()

# %% [markdown]
# ## Inferential: does rainfall differ between rice and chickpea? (Welch t-test + 95% CI) and across all crops (ANOVA)

# %%
rice, chick = (crop.loc[crop.label == c, "rainfall"] for c in ("rice", "chickpea"))
t = stats.ttest_ind(rice, chick, equal_var=False)
ci = t.confidence_interval(0.95)
report["ttest_rice_vs_chickpea_rainfall"] = {
    "mean_rice": round(rice.mean(), 2), "mean_chickpea": round(chick.mean(), 2),
    "t": round(t.statistic, 2), "p": float(t.pvalue),
    "ci95_mean_difference": [round(ci.low, 2), round(ci.high, 2)],
}
anova = stats.f_oneway(*(g.rainfall.values for _, g in crop.groupby("label")))
report["anova_rainfall_across_crops"] = {"F": round(anova.statistic, 1), "p": float(anova.pvalue)}

# %% [markdown]
# ## Yield data

# %%
y = pd.read_csv(RAW / "yield_df.csv")
y = y.drop(columns=[c for c in y.columns if c.startswith("Unnamed")])  # leftover index column
report["yield_raw"] = {"rows": len(y), "missing": int(y.isna().sum().sum()), "duplicates": int(y.duplicated().sum())}
y = y.drop_duplicates().dropna()
y = y[(y["hg/ha_yield"] > 0) & (y.average_rain_fall_mm_per_year >= 0) & (y.pesticides_tonnes >= 0)]
y.to_csv(OUT / "yield_clean.csv", index=False)
report["yield_clean_rows"] = len(y)
report["yield_items"] = y.Item.value_counts().to_dict()
report["yield_describe"] = y.describe().round(2).to_dict()

sns.heatmap(y.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Yield data correlation"); plt.tight_layout(); plt.savefig(FIG / "yield_correlation.png", dpi=120); plt.close()

(ROOT / "source" / "models" / "eda.json").write_text(json.dumps(report, indent=2, default=str))
print(json.dumps({k: report[k] for k in ("crop_raw", "crop_clean_rows", "yield_raw", "yield_clean_rows",
                                         "ttest_rice_vs_chickpea_rainfall", "anova_rainfall_across_crops")}, indent=2))
