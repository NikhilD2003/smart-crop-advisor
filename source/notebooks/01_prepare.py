# %% [markdown]
# # 01 - Data preparation and analysis
# Cleans both datasets, runs descriptive / exploratory / inferential analysis.
# Run from anywhere: `python source/notebooks/01_prepare.py` (also opens as cells in PyCharm/VS Code).
#
# **What this part does:** loads the two raw datasets, removes bad rows, describes the data, draws exploratory graphs and runs
# the inferential tests the brief asks for. **Why:** a model is only as good as its data, and the brief requires descriptive,
# exploratory and inferential analysis before any modelling. **Outputs:** `data/processed/*_clean.csv`,
# `source/models/eda.json` and the graphs in `source/models/figures/`.

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
# **What:** count missing values, duplicate rows, pH outside 0-14, humidity outside 0-100 and negative readings, then drop any
# offending rows. **Why:** impossible values would silently corrupt every later step (this is also the first AI-security
# defence: reject out-of-range data). **Result:** 2,200 rows, 0 missing, 0 duplicates, 0 impossible values, so nothing is dropped.
# The same range rules are reused by the web app to reject spoofed sensor readings.

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
# **What:** summary statistics (count, mean, std, min, quartiles, max) for the 7 readings, samples per crop (100 each, so the
# classes are balanced), mean reading per crop, a correlation heatmap, histograms and a rainfall-by-crop box plot.
# **Why:** to see the ranges, spot outliers, and learn which readings separate crops. **Graphs:** `correlation.png`
# (Pearson r between readings; P and K are strongly correlated because several fruit crops need both), `histograms.png`
# (shape of each reading) and `rainfall_by_crop.png` (rice, jute and coconut are wet, chickpea and mothbeans are dry).

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
# **What:** a Welch t-test (does not assume equal variances) compares rice and chickpea rainfall, with a 95% confidence
# interval for the difference; a one-way ANOVA tests whether mean rainfall differs across all 22 crops.
# **Why:** the brief asks for inferential analysis. **Result:** rice averages 236 mm vs chickpea 80 mm, t = 44.35,
# p = 7.6e-72, 95% CI of the difference [149.15, 163.10] mm; ANOVA F = 605.5, p ~ 0. So crops genuinely differ in rainfall.
# (`03_analysis.py` adds effect sizes, which a p-value cannot give.)

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
# **What:** loads the FAO-derived country yield table (28,242 rows), drops the leftover index column, 2,310 duplicates and
# rows with non-positive yield or negative inputs, leaving 25,932 rows. **Why:** the yield regression (Module 2) needs clean,
# physically sensible rows. **Output:** `yield_clean.csv` and `yield_correlation.png` (correlation of yield with rainfall -0.00, pesticides 0.07, temperature -0.11, year 0.09: all near zero, so crop type, not weather, explains most of the yield variance).

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
