# %% [markdown]
# # 02 - Train, compare, explain, export
# Modules 1-4, 6-8. Needs `01_prepare.py` to have run. Writes metrics + figures to source/models
# and the browser model to frontend/data/model.json.
#
# **What this part does:** trains and compares the classifiers (Modules 1, 3), explains them (Module 6), clusters farms
# (Module 4), fits the yield regression (Module 2), measures Edge-AI size/speed (Module 7), runs the label-poisoning demo
# (Module 8), trains the suitability model, and exports everything for the web app. **Why:** these are the course's ML
# topics; each is evaluated on data the model never saw. **Outputs:** `metrics.json`, `.joblib` models, figures, `model.json`.

# %%
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report, mean_absolute_error,
                             mean_squared_error, r2_score, silhouette_score, homogeneity_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore", message="X does not have valid feature names")
SEED = 42
ROOT = Path(__file__).resolve().parents[2]
PROC, MODELS = ROOT / "data" / "processed", ROOT / "source" / "models"
FIG = MODELS / "figures"
WEB = ROOT / "frontend" / "data"
WEB.mkdir(parents=True, exist_ok=True)
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
M = {}  # all metrics land here

# %% [markdown]
# ## Module 1 - decision tree (depth chosen by cross-validation) vs random forest
# **What:** 80/20 stratified split (every crop in both sets, fixed seed 42). Tree depth 3-12 is compared by 5-fold CV on the
# training set only; the best depth (11) is kept so the tree is not just memorising (overfitting control). A 30-tree random
# forest is trained for comparison. **Result:** tree 97.7%, forest 99.1% test accuracy. **Graphs:** `confusion_rf.png`
# (rows = true crop, columns = predicted; a bright diagonal means correct), `tree_top.png` (first 3 levels of the tree = readable
# rules), `feature_importance.png` (how much each reading reduces impurity across the forest; rainfall and humidity lead).

# %%
crop = pd.read_csv(PROC / "crop_clean.csv")
X, y = crop[FEATURES], crop.label
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)  # stratified 80/20

cv = {d: cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=SEED), Xtr, ytr, cv=5).mean()
      for d in range(3, 13)}
depth = max(cv, key=cv.get)
M["tree_depth_cv"] = {str(d): round(s, 4) for d, s in cv.items()}
M["tree_depth_chosen"] = depth

t0 = time.time(); dt = DecisionTreeClassifier(max_depth=depth, random_state=SEED).fit(Xtr, ytr); dt_time = time.time() - t0
t0 = time.time(); rf = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=SEED).fit(Xtr, ytr); rf_time = time.time() - t0
M["decision_tree"] = {"accuracy": round(accuracy_score(yte, dt.predict(Xte)), 4), "train_seconds": round(dt_time, 3),
                      "train_accuracy": round(dt.score(Xtr, ytr), 4)}
M["random_forest"] = {"accuracy": round(accuracy_score(yte, rf.predict(Xte)), 4), "train_seconds": round(rf_time, 3),
                      "train_accuracy": round(rf.score(Xtr, ytr), 4)}
M["rf_per_crop"] = classification_report(yte, rf.predict(Xte), output_dict=True, zero_division=0)

ConfusionMatrixDisplay.from_predictions(yte, rf.predict(Xte), xticks_rotation=80, colorbar=False,
                                        ax=plt.subplots(figsize=(11, 10))[1])
plt.title("Random forest - confusion matrix (test set)"); plt.tight_layout()
plt.savefig(FIG / "confusion_rf.png", dpi=110); plt.close()

plt.figure(figsize=(18, 8))
plot_tree(dt, feature_names=FEATURES, class_names=list(dt.classes_), filled=True, max_depth=3, fontsize=7)
plt.savefig(FIG / "tree_top.png", dpi=110); plt.close()

imp = pd.Series(rf.feature_importances_, FEATURES).sort_values()
imp.plot.barh(title="Random forest feature importance"); plt.tight_layout()
plt.savefig(FIG / "feature_importance.png", dpi=110); plt.close()
M["feature_importance"] = imp.round(4).to_dict()

# %% [markdown]
# ## Module 3 - neural network (scaler fit on the training set only)
# **What:** a feed-forward network, 7 inputs -> 64 -> 32 hidden units (ReLU) -> 22 softmax outputs. Inputs are standardised
# with a scaler fitted on the training set only (fitting on all data would leak test information). **Result:** 99.1% accuracy,
# 408 epochs, about 2 s training. **Graph:** `mlp_loss.png`, the training loss per epoch; a smooth fall to near zero means it converged.

# %%
scaler = StandardScaler().fit(Xtr)
t0 = time.time()
mlp = MLPClassifier((64, 32), activation="relu", max_iter=500, random_state=SEED).fit(scaler.transform(Xtr), ytr)
mlp_time = time.time() - t0
M["neural_net"] = {"accuracy": round(mlp.score(scaler.transform(Xte), yte), 4), "train_seconds": round(mlp_time, 3),
                   "train_accuracy": round(mlp.score(scaler.transform(Xtr), ytr), 4), "epochs": mlp.n_iter_}
plt.plot(mlp.loss_curve_); plt.xlabel("epoch"); plt.ylabel("training loss"); plt.title("MLP loss curve")
plt.tight_layout(); plt.savefig(FIG / "mlp_loss.png", dpi=110); plt.close()

# %% [markdown]
# ## Module 6 - SHAP on the random forest (offline; the web demo explains with per-reading score cost instead)
# **What:** SHAP assigns each reading a share of credit for each prediction (Shapley values from game theory); the mean of the
# absolute values over 200 test fields ranks the readings. **Why:** explainable AI. **Graph:** `shap_importance.png`
# (humidity, K and P matter most, pH least, which agrees with the ANOVA effect sizes in part 03; the forest's own importance ranks rainfall first, so different explainers weigh correlated readings differently).

# %%
try:
    import shap
    sv = shap.TreeExplainer(rf).shap_values(Xte.iloc[:200])
    sv = np.abs(np.array(sv))
    mean_abs = sv.mean(axis=tuple(i for i in range(sv.ndim) if sv.shape[i] != len(FEATURES)))
    M["shap_mean_abs"] = dict(zip(FEATURES, np.round(mean_abs, 4).tolist()))
    pd.Series(mean_abs, FEATURES).sort_values().plot.barh(title="Mean |SHAP| (random forest)")
    plt.tight_layout(); plt.savefig(FIG / "shap_importance.png", dpi=110); plt.close()
except Exception as e:  # shap is optional
    M["shap_mean_abs"] = f"skipped: {e}"

# %% [markdown]
# ## Module 4 - K-means zones (labels not used for fitting)
# **What:** standardise the 7 readings, run K-means for k = 2..10, and choose k by the silhouette score (restricted to k >= 4
# because k = 2 or 3 gives zones too coarse to describe in words). Each zone is described by its centre (readings more than
# 0.5 standard deviations from average become words such as "hot, humid, high rainfall"). **Result:** k = 10, silhouette
# 0.349, homogeneity vs the true crops 0.64. **Graphs:** `kmeans_k.png` (elbow = inertia, silhouette per k) and
# `kmeans_pca.png` (the zones drawn in 2-D after PCA).

# %%
Xs_all = StandardScaler().fit(X)
Z = Xs_all.transform(X)
ks = range(2, 11)  # plotted for the elbow; k is picked from 4+ because k=2/3 gives zones too coarse to describe
inertia, sil = [], []
for k in ks:
    km = KMeans(k, n_init=10, random_state=SEED).fit(Z)
    inertia.append(km.inertia_); sil.append(silhouette_score(Z, km.labels_))
k_best = max((k for k in ks if k >= 4), key=lambda k: sil[list(ks).index(k)])
km = KMeans(k_best, n_init=10, random_state=SEED).fit(Z)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow (inertia)")
ax[1].plot(ks, sil, "o-"); ax[1].set_title("Silhouette")
plt.tight_layout(); plt.savefig(FIG / "kmeans_k.png", dpi=110); plt.close()
P = PCA(2, random_state=SEED).fit_transform(Z)
plt.scatter(P[:, 0], P[:, 1], c=km.labels_, cmap="tab10", s=8); plt.title(f"K-means zones (k={k_best}) in PCA space")
plt.tight_layout(); plt.savefig(FIG / "kmeans_pca.png", dpi=110); plt.close()

WORDS = {"N": ("low nitrogen", "high nitrogen"), "P": ("low phosphorus", "high phosphorus"),
         "K": ("low potassium", "high potassium"), "temperature": ("cool", "hot"),
         "humidity": ("dry air", "humid"), "ph": ("acidic soil", "alkaline soil"),
         "rainfall": ("low rainfall", "high rainfall")}
zones = []
for i, c in enumerate(km.cluster_centers_):
    traits = [WORDS[f][c[j] > 0] for j, f in enumerate(FEATURES) if abs(c[j]) >= 0.5]
    crops = crop.label[km.labels_ == i].value_counts()
    zones.append({"id": i, "size": int((km.labels_ == i).sum()), "center": [round(float(v), 4) for v in c],
                  "description": ", ".join(traits) or "close to average conditions",
                  "top_crops": crops.head(4).index.tolist()})
M["kmeans"] = {"k": k_best, "silhouette_by_k": {str(k): round(s, 4) for k, s in zip(ks, sil)},
               "homogeneity_vs_crop": round(homogeneity_score(y, km.labels_), 4), "zones": zones}

# %% [markdown]
# ## Module 2 - yield regression (linear + ridge, one-hot crop type); random forest only as a reference
# **What:** predict yield (hg/ha) from crop type (one-hot encoded), annual rainfall, pesticides and average temperature, with
# linear and Ridge regression; a random forest is a reference, not shipped. **Metrics:** MAE (average error), RMSE (penalises big
# errors) and R2 (share of variance explained). **Result:** linear R2 0.638, MAE 32,040 hg/ha (= 3.2 t/ha), RMSE 51,004;
# Ridge is identical; the forest reaches R2 0.967 but is too large for the browser and shows the data has non-linear structure.

# %%
yd = pd.read_csv(PROC / "yield_clean.csv")
YF = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
Ytr, Yte = train_test_split(yd, test_size=0.2, stratify=yd.Item, random_state=SEED)
prep = ColumnTransformer([("item", OneHotEncoder(handle_unknown="ignore"), ["Item"]), ("num", "passthrough", YF)])
M["yield"] = {}
fitted = {}
for name, est in [("linear", LinearRegression()), ("ridge", Ridge(alpha=1.0)),
                  ("random_forest_reference", RandomForestRegressor(100, max_depth=14, n_jobs=-1, random_state=SEED))]:
    pipe = make_pipeline(prep, est).fit(Ytr[["Item"] + YF], Ytr["hg/ha_yield"])
    p = pipe.predict(Yte[["Item"] + YF])
    M["yield"][name] = {"MAE": round(mean_absolute_error(Yte["hg/ha_yield"], p)),
                        "RMSE": round(mean_squared_error(Yte["hg/ha_yield"], p) ** 0.5),
                        "R2": round(r2_score(Yte["hg/ha_yield"], p), 4)}
    fitted[name] = pipe
lin = fitted["linear"]
items = lin[0].named_transformers_["item"].categories_[0].tolist()
coef = lin[-1].coef_.tolist()

# %% [markdown]
# ## Module 7 - Edge AI numbers; Module 8 - label-poisoning demo
# **Edge AI (Module 7):** each model is saved with joblib and timed: tree 15.7 KB / 0.07 ms per prediction, forest 773 KB / 1.2 ms,
# network 121 KB / 0.16 ms. Small and fast means it can run on a low-cost device with no internet.
# **Label poisoning (Module 8):** a share of training labels (0-30%) is replaced with random wrong crops and the tree is retrained.
# **Result:** tree accuracy falls 97.7% -> 95.5% -> 92.5% -> 81.1% at 0, 5, 10, 20% poisoned (82.5% at 30%, a noisy tail),
# showing why training data must come from trusted sources.

# %%
def predict_ms(fn, n=200):
    t0 = time.perf_counter()
    for row in Xte.values[:n]:
        fn(row.reshape(1, -1))
    return round((time.perf_counter() - t0) / n * 1000, 3)

for name, mdl in [("decision_tree", dt), ("random_forest", rf), ("neural_net", mlp)]:
    f = MODELS / f"{name}.joblib"
    joblib.dump(mdl, f)
    M[name]["file_kb"] = round(f.stat().st_size / 1024, 1)
M["decision_tree"]["predict_ms"] = predict_ms(dt.predict)
M["random_forest"]["predict_ms"] = predict_ms(rf.predict)
M["neural_net"]["predict_ms"] = predict_ms(lambda r: mlp.predict(scaler.transform(r)))

rng = np.random.default_rng(SEED)
poison = {}
for frac in (0, 0.05, 0.10, 0.20, 0.30):
    yp = ytr.copy().values
    idx = rng.choice(len(yp), int(frac * len(yp)), replace=False)
    yp[idx] = rng.choice(np.unique(yp), len(idx))  # random wrong labels
    poison[str(frac)] = round(DecisionTreeClassifier(max_depth=depth, random_state=SEED).fit(Xtr, yp).score(Xte, yte), 4)
M["label_poisoning_tree_accuracy"] = poison

# %% [markdown]
# ## Suitability model (trained on the synthetic crop dataset)
# The classifiers above only learn from good fields. This model learns from the generated weak fields too.
# Input = how far each reading is from the crop's usual profile (absolute distance, in spreads), output = suitability 0-1.
# Gradient boosting with a **monotonic constraint**: a reading moving further from typical can never raise the score,
# so improving a reading never lowers it (a plain neural network did not guarantee this).
# Caveat for the report: the labels come from our own distance formula (01b_synthetic.py), so R2 below shows the model
# reproduces that rule. The independent check is `argmax_matches_real_label`: on real held-out fields, does the
# highest-suitability crop equal the true crop?

# %%
from sklearn.ensemble import HistGradientBoostingRegressor

syn = pd.read_csv(PROC / "synthetic_crop_dataset.csv")
profile = json.loads((PROC / "crop_profile.json").read_text())
zfeat = lambda c, x: np.abs(np.clip((np.asarray(x, float) - profile[c]["med"]) / profile[c]["scale"], -6, 6))
Zs = np.array([zfeat(c, r) for c, r in zip(syn.crop, syn[FEATURES].values)])
Str, Ste, str_, ste = train_test_split(Zs, syn.suitability.values, test_size=0.2, stratify=syn.crop, random_state=SEED)
sreg = HistGradientBoostingRegressor(monotonic_cst=[-1] * 7, max_iter=300, max_leaf_nodes=12, learning_rate=0.1,
                                     random_state=SEED).fit(Str, str_)
clip = lambda a: np.clip(a, 0, 1)
M["suitability"] = {"MAE": round(mean_absolute_error(ste, clip(sreg.predict(Ste))), 4),
                    "R2": round(r2_score(ste, clip(sreg.predict(Ste))), 4), "rows": len(syn)}
bands = lambda a: pd.cut(a, [-1, .3, .7, 2], labels=["unsuitable", "marginal", "suitable"])
M["suitability"]["tier_accuracy"] = round(float((bands(ste) == bands(clip(sreg.predict(Ste)))).mean()), 4)
M["suitability"]["unsuitable_predicted_under_30pct"] = round(float((clip(sreg.predict(Ste))[ste < .3] < .3).mean()), 4)
# independent check on REAL held-out fields (not generated): highest-suitability crop vs the true crop
cls = rf.classes_.tolist()
suit_all = lambda x: [float(clip(sreg.predict([zfeat(c, x)]))[0]) for c in cls]
hit = [cls[int(np.argmax(suit_all(x)))] == t for x, t in zip(Xte.values, yte)]
M["suitability"]["argmax_matches_real_label"] = round(float(np.mean(hit)), 4)
print(M["suitability"])

# %% [markdown]
# ## Export for the browser (Edge AI: everything runs client-side)

# %%
def tree_json(est):
    t = est.tree_
    val = t.value[:, 0, :]
    val = val / val.sum(1, keepdims=True)
    v = [None if t.children_left[i] != -1 else
         [x for c, p in enumerate(val[i]) if p >= 0.005 for x in (c, round(float(p), 3))]
         for i in range(t.node_count)]
    return {"f": t.feature.tolist(), "t": [round(float(x), 6) for x in t.threshold],
            "l": t.children_left.tolist(), "r": t.children_right.tolist(), "v": v}

stats = {c: {f: [round(float(np.percentile(g[f], q)), 2) for q in (10, 50, 90)] for f in FEATURES}
         for c, g in crop.groupby("label")}  # [p10, median, p90] per crop - the rule engine's thresholds
yield_for_crop = {"maize": "Maize", "rice": "Rice, paddy"}
assert all(v in items for v in yield_for_crop.values())

model = {
    "features": FEATURES, "classes": rf.classes_.tolist(),
    "ranges": {f: [float(X[f].min()), float(X[f].max())] for f in FEATURES},
    "crop_stats": stats,
    "tree": tree_json(dt), "forest": [tree_json(e) for e in rf.estimators_],
    "importance": M["feature_importance"],
    "mlp": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "W": [np.round(w, 6).tolist() for w in mlp.coefs_], "b": [np.round(b, 6).tolist() for b in mlp.intercepts_]},
    "kmeans": {"mean": Xs_all.mean_.tolist(), "scale": Xs_all.scale_.tolist(), "zones": zones},
    "yield": {"items": items, "intercept": float(lin[-1].intercept_), "item_coef": coef[:len(items)],
              "num_features": YF, "num_coef": coef[len(items):],
              "ranges": {f: [float(yd[f].min()), float(yd[f].max())] for f in YF},
              "median_pesticides": float(yd.pesticides_tonnes.median()),
              "for_crop": yield_for_crop, "metrics": M["yield"]["linear"]},
    "suit": {"profile": profile, "base": float(sreg._baseline_prediction.ravel()[0]), "metrics": M["suitability"],
             "trees": [{"f": n["feature_idx"].tolist(), "t": [round(float(v), 6) for v in n["num_threshold"]],
                        "l": n["left"].tolist(), "r": n["right"].tolist(), "v": [round(float(v), 7) for v in n["value"]],
                        "leaf": n["is_leaf"].astype(int).tolist()} for n in (p[0].nodes for p in sreg._predictors)]},
    "metrics": {k: M[k] for k in ("decision_tree", "random_forest", "neural_net")},
}
out = WEB / "model.json"
out.write_text(json.dumps(model, separators=(",", ":")))
M["web_model_kb"] = round(out.stat().st_size / 1024, 1)

# parity cases: the JS port must reproduce these (see source/tests/parity.mjs)
cases = Xte.iloc[:150]
(ROOT / "source" / "tests").mkdir(exist_ok=True)
(ROOT / "source" / "tests" / "parity_cases.json").write_text(json.dumps({
    "x": cases.values.tolist(), "tree": dt.predict(cases).tolist(), "forest": rf.predict(cases).tolist(),
    "mlp": mlp.predict(scaler.transform(cases)).tolist(),
    "zone": km.predict(Xs_all.transform(cases)).tolist(),
    "suit": [[c, [round(float(v), 4) for v in x], round(float(clip(sreg.predict([zfeat(c, x)]))[0]), 4)]
             for c, x in zip(["rice", "apple", "lentil", "mothbeans", "coffee"] * 6, Xte.iloc[:30].values)]}))

(MODELS / "metrics.json").write_text(json.dumps(M, indent=2, default=str))
print(json.dumps({k: M[k] for k in ("tree_depth_chosen", "decision_tree", "random_forest", "neural_net", "yield",
                                    "label_poisoning_tree_accuracy", "web_model_kb")}, indent=2))
print("kmeans k =", k_best, "homogeneity", M["kmeans"]["homogeneity_vs_crop"])
for z in zones: print(z["id"], z["size"], z["description"], z["top_crops"])
