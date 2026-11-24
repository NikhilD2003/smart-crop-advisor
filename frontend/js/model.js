// Browser-side inference (Edge AI): decision tree, random forest, MLP, K-means zones, yield regression.
// DOM-free so source/tests/parity.mjs can run it in Node against the Python models.

const argmax = (a) => a.reduce((b, v, i) => (v > a[b] ? i : b), 0);
const f32 = Math.fround; // sklearn compares float32 inputs with the thresholds

function leaf(t, x) {
  let n = 0;
  while (t.l[n] !== -1) n = f32(x[t.f[n]]) <= t.t[n] ? t.l[n] : t.r[n];
  return t.v[n]; // sparse [class, prob, class, prob, ...]
}

export function forestProba(m, x) {
  const p = new Array(m.classes.length).fill(0);
  for (const t of m.forest) {
    const v = leaf(t, x);
    for (let i = 0; i < v.length; i += 2) p[v[i]] += v[i + 1];
  }
  return p.map((s) => s / m.forest.length);
}

export function treePredict(m, x) {
  const v = leaf(m.tree, x);
  let best = 0;
  for (let i = 0; i < v.length; i += 2) if (v[i + 1] > (v[best + 1] ?? 0) || i === 0) best = i;
  return v[best];
}

export function mlpProba(m, x) {
  const { mean, scale, W, b } = m.mlp;
  let a = x.map((v, i) => (v - mean[i]) / scale[i]);
  W.forEach((w, l) => {
    const out = b[l].map((bias, j) => bias + a.reduce((s, v, k) => s + v * w[k][j], 0));
    a = l < W.length - 1 ? out.map((v) => Math.max(0, v)) : out;
  });
  const mx = Math.max(...a), e = a.map((v) => Math.exp(v - mx)), s = e.reduce((p, v) => p + v, 0);
  return e.map((v) => v / s);
}

export function zone(m, x) {
  const { mean, scale, zones } = m.kmeans;
  const z = x.map((v, i) => (v - mean[i]) / scale[i]);
  const d = zones.map((c) => c.center.reduce((s, v, i) => s + (v - z[i]) ** 2, 0));
  return zones[argmax(d.map((v) => -v))];
}

// Explainable AI: how much does the winning crop's probability drop when one reading is replaced by the dataset average?
export function explain(m, x, classIdx) {
  const base = forestProba(m, x)[classIdx];
  return m.features
    .map((f, i) => {
      const y = x.slice(); y[i] = m.kmeans.mean[i];
      return { feature: f, drop: base - forestProba(m, y)[classIdx] };
    })
    .sort((a, b) => b.drop - a.drop);
}

export function predictYield(m, item, rain, pesticides, temp) {
  const y = m.yield, i = y.items.indexOf(item);
  const hg = y.intercept + y.item_coef[i] + y.num_coef[0] * rain + y.num_coef[1] * pesticides + y.num_coef[2] * temp;
  return Math.max(0, hg) / 10000; // hg/ha -> t/ha
}

// Input validation (AI security): hard = physically impossible, warn = outside the training data.
const HARD = { N: [0, 300], P: [0, 300], K: [0, 300], temperature: [-10, 60], humidity: [0, 100], ph: [0, 14], rainfall: [0, 1000] };
export function validate(m, x) {
  const errors = [], warnings = [];
  m.features.forEach((f, i) => {
    const v = x[i];
    if (!Number.isFinite(v)) return errors.push(`${f} is missing`);
    if (v < HARD[f][0] || v > HARD[f][1]) return errors.push(`${f} = ${v} is outside the possible range ${HARD[f][0]}-${HARD[f][1]}`);
    const [lo, hi] = m.ranges[f];
    if (v < lo || v > hi) warnings.push(`${f} = ${v} is outside the range the model was trained on (${+lo.toFixed(1)}-${+hi.toFixed(1)}); treat the result with extra caution`);
  });
  return { errors, warnings };
}

// Suitability 0-1 of one crop for these readings: monotone boosted trees on |distance from the crop's usual profile|,
// trained on the synthetic crop dataset. Further from typical never scores higher.
export function suitability(m, crop, x) {
  const { med, scale } = m.suit.profile[crop];
  const z = x.map((v, i) => Math.abs(Math.max(-6, Math.min(6, (v - med[i]) / scale[i]))));
  let s = m.suit.base;
  for (const t of m.suit.trees) {
    let n = 0;
    while (!t.leaf[n]) n = z[t.f[n]] <= t.t[n] ? t.l[n] : t.r[n];
    s += t.v[n];
  }
  return Math.max(0, Math.min(1, s));
}
