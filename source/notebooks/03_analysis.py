# %% [markdown]
# # 03 - Ablation study, statistical analysis and cross-dataset validation
# **Why this notebook part exists.** Part 02 reports one accuracy number per model. That raises three fair questions:
# 1. *Which parts of the system actually matter?* -> **Ablation study** (remove a part, measure what breaks).
# 2. *Are the differences between models real or just luck on a small test set?* -> **Statistical analysis**
#    (confidence intervals, significance tests, effect sizes).
# 3. *Does it still work on data it was not built from?* -> **Cross-dataset validation**.
#
# Needs `01_prepare.py`, `01b_synthetic.py` and `02_train.py` to have run. Writes `source/models/analysis.json`
# and the figures `stat_*.png`, `ablation_*.png`, `suitability_*.png`, `cross_dataset.png`.

# %%
import json
import urllib.request
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42
ROOT = Path(__file__).resolve().parents[2]
PROC, MODELS = ROOT / "data" / "processed", ROOT / "source" / "models"
FIG = MODELS / "figures"
EXT = ROOT / "data" / "external"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
A = {}  # everything measured here lands in analysis.json

crop = pd.read_csv(PROC / "crop_clean.csv")
X, y = crop[FEATURES], crop.label
# the SAME stratified 80/20 split as 02_train.py (same seed), so numbers are comparable
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
depth = json.loads((MODELS / "metrics.json").read_text())["tree_depth_chosen"]
mk_dt = lambda: DecisionTreeClassifier(max_depth=depth, random_state=SEED)
mk_rf = lambda: RandomForestClassifier(n_estimators=30, max_depth=10, random_state=SEED)
mk_mlp = lambda: make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=500, random_state=SEED))
models = {"Decision tree": mk_dt().fit(Xtr, ytr), "Random forest": mk_rf().fit(Xtr, ytr), "Neural network": mk_mlp().fit(Xtr, ytr)}
pred = {k: m.predict(Xte) for k, m in models.items()}
print({k: round(float((p == yte.values).mean()), 4) for k, p in pred.items()})

# %% [markdown]
# # Part A - Statistical analysis
# ## A1. Bootstrap 95% confidence interval for each model's test accuracy
# **What:** the test set has only 440 fields, so "99.1%" is an estimate with uncertainty. We resample the 440 test
# predictions *with replacement* 2,000 times, recompute accuracy each time, and take the 2.5th and 97.5th percentiles.
# **Why:** if two models' intervals overlap heavily, we cannot claim one is better. **Output:** the bar chart below.

# %%
rng = np.random.default_rng(SEED)
correct = {k: (p == yte.values).astype(float) for k, p in pred.items()}
n = len(yte)
boot = {k: [c[rng.integers(0, n, n)].mean() for _ in range(2000)] for k, c in correct.items()}
A["bootstrap_accuracy_ci95"] = {k: {"accuracy": round(float(correct[k].mean()), 4),
                                    "ci95": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]}
                                for k, b in boot.items()}
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for i, (k, v) in enumerate(A["bootstrap_accuracy_ci95"].items()):
    ax.errorbar(v["accuracy"], i, xerr=[[v["accuracy"] - v["ci95"][0]], [v["ci95"][1] - v["accuracy"]]], fmt="o", capsize=6)
    ax.text(v["ci95"][1] + 0.002, i, f'{v["accuracy"]:.3f}  [{v["ci95"][0]:.3f}, {v["ci95"][1]:.3f}]', va="center", fontsize=8)
ax.set_yticks(range(3)); ax.set_yticklabels(A["bootstrap_accuracy_ci95"]); ax.set_xlim(0.94, 1.03)
ax.set_xlabel("test accuracy (point) with bootstrap 95% CI"); ax.set_title("A1. Are the models really different?")
plt.tight_layout(); plt.savefig(FIG / "stat_bootstrap_ci.png", dpi=120); plt.show()
print(json.dumps(A["bootstrap_accuracy_ci95"], indent=1))

# %% [markdown]
# ## A2. McNemar's exact test - do two models make different mistakes on the same test fields?
# **What:** for a pair of models, count fields where only model 1 is right (b) and only model 2 is right (c). Under "the
# models are equally good", b and c should be about equal; an exact binomial test on b vs b+c gives the p-value.
# **Why:** it is the standard paired test for two classifiers on one test set. p < 0.05 means a real difference.

# %%
names = list(models)
A["mcnemar_exact"] = {}
for i in range(3):
    for j in range(i + 1, 3):
        a, b_ = correct[names[i]], correct[names[j]]
        only_a, only_b = int(((a == 1) & (b_ == 0)).sum()), int(((a == 0) & (b_ == 1)).sum())
        p = 1.0 if only_a + only_b == 0 else float(stats.binomtest(only_a, only_a + only_b, 0.5).pvalue)
        A["mcnemar_exact"][f"{names[i]} vs {names[j]}"] = {"only_first_right": only_a, "only_second_right": only_b,
                                                             "p": round(p, 6), "significant_at_5pct": p < 0.05}
print(json.dumps(A["mcnemar_exact"], indent=1))

# %% [markdown]
# ## A3. Repeated 5-fold cross-validation + paired tests
# **What:** instead of one split, train each model on 15 different train/test splits (5 folds x 3 repeats) and compare the
# 15 paired accuracies with a paired t-test and a Wilcoxon signed-rank test.
# **Caveat:** CV folds share training data, so the p-values are optimistic; read them together with A1/A2.

# %%
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
scores = {k: cross_val_score(mk(), X, y, cv=cv, n_jobs=-1) for k, mk in
          [("Decision tree", mk_dt), ("Random forest", mk_rf), ("Neural network", mk_mlp)]}
A["repeated_cv_accuracy"] = {k: {"mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4)} for k, s in scores.items()}
A["paired_tests_on_cv"] = {}
for i in range(3):
    for j in range(i + 1, 3):
        a, b_ = scores[names[i]], scores[names[j]]
        t = stats.ttest_rel(a, b_)
        w = stats.wilcoxon(a, b_) if np.any(a != b_) else None
        A["paired_tests_on_cv"][f"{names[i]} vs {names[j]}"] = {
            "mean_diff": round(float((a - b_).mean()), 4), "paired_t_p": round(float(t.pvalue), 6),
            "wilcoxon_p": None if w is None else round(float(w.pvalue), 6)}
print(json.dumps({k: A[k] for k in ("repeated_cv_accuracy", "paired_tests_on_cv")}, indent=1))

# %% [markdown]
# ## A4. Effect size: how strongly does each feature separate the 22 crops? (ANOVA eta-squared)
# **What:** for each feature, eta^2 = (variance *between* crops) / (total variance), from a one-way ANOVA with crop as the
# group. 0 = crop tells you nothing about the feature, 1 = crop fully determines it.
# **Why:** 01_prepare.py gave ANOVA p-values (all tiny, because n is large). Effect size says *how much*, which p cannot.
# It also explains the feature importance seen in 02 (rainfall, humidity and K matter most).

# %%
eta2 = {}
for f in FEATURES:
    groups = [g[f].values for _, g in crop.groupby("label")]
    grand = crop[f].mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    sst = ((crop[f] - grand) ** 2).sum()
    eta2[f] = round(float(ssb / sst), 4)
A["anova_eta_squared"] = eta2
pd.Series(eta2).sort_values().plot.barh(figsize=(6, 3.6), color="#3d8f4f")
plt.xlabel("eta squared (share of variance explained by crop)"); plt.title("A4. Which readings separate crops?")
plt.tight_layout(); plt.savefig(FIG / "stat_eta2.png", dpi=120); plt.show()
print(eta2)

# %% [markdown]
# # Part B - Ablation study
# An ablation study removes one component at a time and measures the damage. Components that cost nothing to remove are
# candidates to delete; components whose removal hurts are justified.
# ## B1. Leave-one-feature-out (random forest, 5-fold CV accuracy)
# **What:** retrain the forest without one reading (or one group: nutrients N/P/K, climate temp/humidity/rainfall, pH) and
# compare with the full 7-feature model. **Why:** shows which sensors a farmer really needs. A cheap sensor whose removal
# costs almost nothing could be dropped on a low-cost device.

# %%
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
cvacc = lambda cols: cross_val_score(mk_rf(), X[cols], y, cv=skf, n_jobs=-1)
full = cvacc(FEATURES)
abl = {"all 7 features": full}
for f in FEATURES:
    abl[f"without {f}"] = cvacc([c for c in FEATURES if c != f])
groups = {"without N, P, K (soil nutrients)": ["N", "P", "K"], "without temp, humidity, rainfall (climate)": ["temperature", "humidity", "rainfall"]}
for k, drop in groups.items():
    abl[k] = cvacc([c for c in FEATURES if c not in drop])
A["ablation_features_rf_cv"] = {k: {"mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4),
                                    "change_vs_full": round(float(v.mean() - full.mean()), 4)} for k, v in abl.items()}
s = pd.Series({k: v.mean() for k, v in abl.items()}).sort_values()
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.barh(s.index, s.values, color=["#b5483a" if "all" not in k else "#3d8f4f" for k in s.index])
ax.set_xlim(0.5, 1.0); ax.axvline(full.mean(), color="k", ls="--", lw=1)
for i, v in enumerate(s.values):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
ax.set_xlabel("5-fold CV accuracy"); ax.set_title("B1. Feature ablation (dashed = all features)")
plt.tight_layout(); plt.savefig(FIG / "ablation_features.png", dpi=120); plt.show()
print(pd.DataFrame(A["ablation_features_rf_cv"]).T.round(4).to_string())

# %% [markdown]
# ## B2. Ablating model size (number of trees) - the Edge AI trade-off
# **What:** accuracy of a forest with 1, 3, 5, 10, 30, 100 trees (test set), and its approximate size = total tree nodes.
# **Why:** the web app ships the forest to the browser, so fewer trees means a smaller download. We pick the knee of the
# curve (30 trees).

# %%
ntrees = [1, 3, 5, 10, 30, 100]
res = []
for k in ntrees:
    m = RandomForestClassifier(n_estimators=k, max_depth=10, random_state=SEED).fit(Xtr, ytr)
    res.append({"trees": k, "accuracy": round(float(m.score(Xte, yte)), 4), "nodes": int(sum(e.tree_.node_count for e in m.estimators_))})
A["ablation_forest_size"] = res
fig, ax = plt.subplots(figsize=(6.5, 3.8)); ax2 = ax.twinx()
ax.plot(ntrees, [r["accuracy"] for r in res], "o-", color="#3d8f4f"); ax2.plot(ntrees, [r["nodes"] for r in res], "s--", color="#b5483a")
ax.set_xscale("log"); ax.set_xlabel("number of trees"); ax.set_ylabel("test accuracy", color="#3d8f4f"); ax2.set_ylabel("total nodes (size)", color="#b5483a")
ax.set_title("B2. Accuracy vs size"); plt.tight_layout(); plt.savefig(FIG / "ablation_forest_size.png", dpi=120); plt.show()
print(pd.DataFrame(res).to_string(index=False))

# %% [markdown]
# ## B3. Ablating the suitability model
# The suitability score (see 01b/02) is trained on the **synthetic** crop dataset. Three components are tested, all on the
# *same held-out synthetic rows* as 02:
# * **Full model** - gradient boosting, monotone constraint, all row types.
# * **No monotone constraint** - same model but free to bend. Measured by *monotone violations*: take a random field, push one
#   reading 1.5x further from the crop's typical value, and count how often the score wrongly *rises*.
# * **Only good + other-crop rows** - trained without the marginal examples (blends, one-reading-off, single-extreme rows).
# * **Classifier only (no suitability model)** - use the random forest's probability for that crop as the score.
#
# Metrics: R2 and MAE against the synthetic label, violations, and the independent check `argmax_matches_real_label`
# (on **real** held-out fields, is the top-scoring crop the true crop?).

# %%
syn = pd.read_csv(PROC / "synthetic_crop_dataset.csv")
profile = json.loads((PROC / "crop_profile.json").read_text())
cls = sorted(crop.label.unique())
med = {c: np.array(profile[c]["med"]) for c in cls}; sc = {c: np.array(profile[c]["scale"]) for c in cls}
zabs = lambda c, Xm: np.abs(np.clip((np.asarray(Xm, float) - med[c]) / sc[c], -6, 6))
idx_tr, idx_te = train_test_split(np.arange(len(syn)), test_size=0.2, stratify=syn.crop, random_state=SEED)
Zall = np.array([zabs(c, r) for c, r in zip(syn.crop, syn[FEATURES].values)])


def fit_hgb(rows, mono=True):
    return HistGradientBoostingRegressor(monotonic_cst=[-1] * 7 if mono else None, max_iter=300, max_leaf_nodes=12,
                                         learning_rate=0.1, random_state=SEED).fit(Zall[rows], syn.suitability.values[rows])


rf_full = models["Random forest"]
variants = {
    "Full model (monotone, all rows)": (lambda c, Xm, m=fit_hgb(idx_tr): np.clip(m.predict(zabs(c, Xm)), 0, 1)),
    "No monotone constraint": (lambda c, Xm, m=fit_hgb(idx_tr, mono=False): np.clip(m.predict(zabs(c, Xm)), 0, 1)),
    "Only good + other-crop rows": (lambda c, Xm, m=fit_hgb(np.array([i for i in idx_tr if syn.how_generated[i] in ("real_good", "other_crop")])):
                                        np.clip(m.predict(zabs(c, Xm)), 0, 1)),
    "Classifier only (forest probability)": (lambda c, Xm: rf_full.predict_proba(np.asarray(Xm, float))[:, list(rf_full.classes_).index(c)]),
}
te = syn.iloc[idx_te]
rrng = np.random.default_rng(7)
viol_cases = []
for _ in range(600):
    c = cls[rrng.integers(len(cls))]; x = med[c] + (rrng.random(7) * 8 - 4) * sc[c]; i = rrng.integers(7)
    y2 = x.copy(); y2[i] = med[c][i] + (x[i] - med[c][i]) * 1.5
    viol_cases.append((c, x, y2))
real_te = Xte.values
abl_s = {}
for name, fn in variants.items():
    pred_s = np.concatenate([fn(c, g[FEATURES].values) for c, g in te.groupby("crop")])
    true_s = np.concatenate([g.suitability.values for _, g in te.groupby("crop")])
    viol = np.mean([float(fn(c, y2.reshape(1, -1))[0] > fn(c, x.reshape(1, -1))[0] + 1e-9) for c, x, y2 in viol_cases])
    top = np.array(cls)[np.argmax(np.column_stack([fn(c, real_te) for c in cls]), axis=1)]
    abl_s[name] = {"R2": round(float(r2_score(true_s, pred_s)), 4), "MAE": round(float(mean_absolute_error(true_s, pred_s)), 4),
                   "monotone_violation_rate": round(float(viol), 4), "argmax_matches_real_label": round(float((top == yte.values).mean()), 4)}
A["ablation_suitability"] = abl_s
df = pd.DataFrame(abl_s).T
print(df.to_string())
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a_, col, ttl in zip(ax, ["R2", "monotone_violation_rate", "argmax_matches_real_label"],
                        ["R2 vs synthetic label (higher better)", "Monotone violations (lower better)", "Top crop = true crop on REAL fields"]):
    a_.barh(range(len(df)), df[col], color="#3d8f4f"); a_.set_yticks(range(len(df)))
    a_.set_yticklabels([s[:30] for s in df.index] if col == "R2" else [""] * len(df)); a_.set_title(ttl, fontsize=9)
    for i, v in enumerate(df[col]):
        a_.text(v, i, f" {v:.3f}", va="center", fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "ablation_suitability.png", dpi=120); plt.show()

# %% [markdown]
# ### B3 figure: the score as one reading improves
# Rice, rainfall swept from 20 to 300 mm with every other reading fixed at rice's typical value. The full model rises
# smoothly and never dips; the unconstrained model can wobble; the classifier-only score jumps because forests are
# piecewise-constant and know nothing about "bad" fields.

# %%
sweep = np.linspace(20, 300, 120)
base = med["rice"].copy()
pts = np.tile(base, (len(sweep), 1)); pts[:, 6] = sweep
plt.figure(figsize=(7.5, 4))
for name, fn in variants.items():
    plt.plot(sweep, fn("rice", pts), label=name, lw=2 if name.startswith("Full") else 1.2)
plt.axhline(0.3, color="grey", ls=":"); plt.axhline(0.7, color="grey", ls=":")
plt.xlabel("rainfall (mm); other readings at rice's typical values"); plt.ylabel("rice suitability"); plt.legend(fontsize=7)
plt.title("Suitability vs rainfall (dotted = 30% / 70% tier lines)"); plt.tight_layout()
plt.savefig(FIG / "suitability_curve.png", dpi=120); plt.show()

# %% [markdown]
# # Part C - Cross-dataset validation
# **What:** test the models on a *different* dataset than the one they were trained on.
# **Dataset used:** `Jakehills/Crop_Yield_Fertilizer` on Hugging Face - 25,000 rows, same 7 features and 22 crops. It was
# checked in this project: 2,199 of its rows are exactly our original rows; the other ~22,800 are **augmented** rows with a
# wider spread (e.g. rice temperature 14-31 C instead of 20-27 C). We call those 22,800 the *external* set.
# **Honest limit:** the external rows are derived from the same Kaggle source, so this tests robustness to a wider,
# shifted distribution, **not** independence from the original source. No fully independent public dataset with these exact
# 7 columns and 22 crops was found.
# **Directions:** (1) train on ours, test on external. (2) train on external, test on our original rows.
# The file is downloaded to `data/external/` (not committed to git); if there is no internet this part is skipped.

# %%
EXT.mkdir(parents=True, exist_ok=True)
f_ext = EXT / "Crop_Yield_Fertilizer.csv"
URL = "https://huggingface.co/datasets/Jakehills/Crop_Yield_Fertilizer/resolve/main/Crop_Yield_Fertilizer.csv"
try:
    if not f_ext.exists():
        urllib.request.urlretrieve(URL, f_ext)
    ext_all = pd.read_csv(f_ext)
    ok = True
except Exception as e:  # offline: skip, do not fail the notebook
    ok = False
    A["cross_dataset"] = f"skipped (could not download): {e}"
    print(A["cross_dataset"])

# %%
if ok:
    key = lambda d: d[FEATURES].astype(float).round(4).astype(str).agg("|".join, axis=1) + "|" + d["label"]
    orig_keys = set(key(crop)); ext_all["k"] = key(ext_all)
    ext = ext_all[~ext_all.k.isin(orig_keys)].reset_index(drop=True)
    Xe, ye = ext[FEATURES], ext.label
    # (1) train on ours (train split), evaluate in-dataset (our test split) and on the external set
    cd = {"external_rows": len(ext), "overlap_with_original_rows": int(len(ext_all) - len(ext))}
    cd["train_ours_test_external"] = {}
    for k, m in models.items():
        cd["train_ours_test_external"][k] = {"in_dataset_accuracy": round(float((m.predict(Xte) == yte.values).mean()), 4),
                                             "external_accuracy": round(float((m.predict(Xe) == ye.values).mean()), 4)}
    sfn = variants["Full model (monotone, all rows)"]
    top_e = np.array(cls)[np.argmax(np.column_stack([sfn(c, Xe.values) for c in cls]), axis=1)]
    cd["train_ours_test_external"]["Suitability (top crop)"] = {
        "in_dataset_accuracy": round(float((np.array(cls)[np.argmax(np.column_stack([sfn(c, Xte.values) for c in cls]), axis=1)] == yte.values).mean()), 4),
        "external_accuracy": round(float((top_e == ye.values).mean()), 4)}
    # (2) train on a stratified 2,200-row sample of external (same size as ours), test on our ORIGINAL rows
    samp = ext.groupby("label", group_keys=False).apply(lambda g: g.sample(100, random_state=SEED))
    cd["train_external_test_ours"] = {}
    for k, mk in [("Decision tree", mk_dt), ("Random forest", mk_rf), ("Neural network", mk_mlp)]:
        m = mk().fit(samp[FEATURES], samp.label)
        cd["train_external_test_ours"][k] = {"accuracy_on_original_rows": round(float((m.predict(X) == y.values).mean()), 4)}
    # distribution shift per feature: two-sample Kolmogorov-Smirnov (0 = same distribution, 1 = completely different)
    cd["ks_shift_per_feature"] = {f: round(float(stats.ks_2samp(crop[f], ext[f]).statistic), 3) for f in FEATURES}
    A["cross_dataset"] = cd
    print(json.dumps(cd, indent=1))

# %% [markdown]
# ### Cross-dataset figure
# Left: accuracy when tested on our own held-out rows vs the external set (trained on ours). A small drop means the model
# generalises to wider conditions; a large drop means it memorised our narrow ranges. Right: how far each feature's
# distribution moved (KS statistic).

# %%
if ok:
    t = pd.DataFrame(cd["train_ours_test_external"]).T
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    xs = np.arange(len(t))
    ax[0].bar(xs - 0.2, t.in_dataset_accuracy, 0.4, label="our test rows", color="#3d8f4f")
    ax[0].bar(xs + 0.2, t.external_accuracy, 0.4, label="external (wider spread)", color="#b5483a")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(t.index, rotation=15, fontsize=8); ax[0].set_ylim(0.5, 1.02); ax[0].legend(fontsize=8)
    for i, (a_, b_) in enumerate(zip(t.in_dataset_accuracy, t.external_accuracy)):
        ax[0].text(i - 0.2, a_, f"{a_:.2f}", ha="center", fontsize=7); ax[0].text(i + 0.2, b_, f"{b_:.2f}", ha="center", fontsize=7)
    ax[0].set_title("C. Train on ours -> test elsewhere")
    pd.Series(cd["ks_shift_per_feature"]).sort_values().plot.barh(ax=ax[1], color="#7a7a7a"); ax[1].set_title("Distribution shift (KS statistic)")
    plt.tight_layout(); plt.savefig(FIG / "cross_dataset.png", dpi=120); plt.show()

# %%
(MODELS / "analysis.json").write_text(json.dumps(A, indent=2, default=str))
print("saved analysis.json with sections:", list(A))
