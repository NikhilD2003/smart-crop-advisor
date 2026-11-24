import { forestProba, treePredict, mlpProba, zone, predictYield, validate, suitability } from "./model.js";
import { deriveFacts, forwardChain, backwardChain } from "./rules.js";
import { reduced, initScroll, intro, scrollReveals, animateResults, pulseCrop } from "./motion.js";
import { startHero } from "./hero3d.js";

const $ = (id) => document.getElementById(id);
const cap = (s) => s[0].toUpperCase() + s.slice(1);
const pct = (v) => (v * 100).toFixed(0) + "%";
const LABEL = { N: "Nitrogen (N)", P: "Phosphorus (P)", K: "Potassium (K)", temperature: "Temperature (°C)",
  humidity: "Humidity (%)", ph: "Soil pH", rainfall: "Rainfall (mm)" };
const STEP = { N: 1, P: 1, K: 1, temperature: 0.1, humidity: 0.1, ph: 0.1, rainfall: 1 };
const argmax = (a) => a.reduce((b, v, i) => (v > a[b] ? i : b), 0);

const [m, rules] = await Promise.all(["data/model.json", "data/rules.json"].map((u) => fetch(u).then((r) => r.json())));

// ---- form ----
$("form").innerHTML = m.features.map((f) => `
  <label>${LABEL[f]}
    <input id="in-${f}" type="number" inputmode="decimal" step="${STEP[f]}" required>
    <small>typical range ${m.ranges[f].map((v) => +v.toFixed(1)).join("–")}</small>
  </label>`).join("");
$("preset").insertAdjacentHTML("beforeend", m.classes.map((c) => `<option>${c}</option>`).join(""));
const fill = (crop) => m.features.forEach((f) => ($(`in-${f}`).value = m.crop_stats[crop][f][1]));
$("preset").onchange = (e) => e.target.value && fill(e.target.value);
fill("rice");
$("preset").value = "rice";
$("go").onclick = () => run(true);
$("form").onsubmit = (e) => { e.preventDefault(); run(true); };

function run(animate = false) {
  const x = m.features.map((f) => parseFloat($(`in-${f}`).value));
  const { errors, warnings } = validate(m, x);
  $("messages").innerHTML = [...errors.map((e) => `<li class="err">${e}</li>`), ...warnings.map((w) => `<li class="warn">${w}</li>`)].join("");
  $("results").hidden = errors.length > 0;
  if (errors.length) return; // reject impossible / spoofed readings instead of advising on them
  render(x, animate);
}

function render(x, animate) {
  const p = forestProba(m, x);
  const S = m.classes.map((c) => suitability(m, c, x)); // 0-1 per crop, trained on good AND weak fields
  const order = S.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
  const best = order[0][1], crop = m.classes[best], conf = S[best];
  const z = zone(m, x);
  // Confidence only says which crop profile is closest. Fit says how many readings are in that crop's usual (10th-90th pct) range.
  const st = m.crop_stats[crop], off = m.features.filter((f, i) => x[i] < st[f][0] || x[i] > st[f][2]);
  const fitOk = off.length <= 1 && conf >= 0.7;

  $("rec").innerHTML = `
    <p class="tag">02 · RESULT</p>
    <h2>Recommended crop</h2>
    <p class="big"><span id="crop-name">${cap(crop)}</span> <span class="pill ${fitOk ? "ok" : "low"}"><span data-count="${(conf * 100).toFixed(0)}" data-suffix="%">${pct(conf)}</span> suitability</span></p>
    <p class="muted"><strong>Field fit: ${m.features.length - off.length} of ${m.features.length} readings</strong> are in the usual range for ${crop}.</p>
    ${conf < 0.3 ? `<p class="warn-text">No crop in our list fits these conditions well. Even the best match (${crop}) has a very low chance of growing here.</p>`
      : conf < 0.7 ? `<p class="warn-text">${cap(crop)} is only a marginal fit${off.length ? `: ${off.map((f) => LABEL[f]).join(", ")} fall outside its usual range` : ""}. Improve those readings and the score rises.</p>` : ""}
    <p class="muted">Other options</p>
    ${order.slice(1, 4).filter(([v]) => v >= 0.01).map(([v, i]) => bar(cap(m.classes[i]), v)).join("")}
    <label class="check">Can I grow a specific crop here?
      <select id="check"><option value="">Pick a crop…</option>${m.classes.map((c) => `<option>${c}</option>`).join("")}</select></label>
    <p id="check-out" class="muted"></p>
    <p class="muted">Your farm sits in climate zone ${z.id + 1}: <strong>${z.description}</strong>. Crops usually grown in this zone: ${z.top_crops.join(", ")}.</p>`;

  $("check").onchange = (e) => {
    const c = e.target.value; if (!c) return ($("check-out").textContent = "");
    const v = S[m.classes.indexOf(c)], rank = order.findIndex(([, i]) => m.classes[i] === c) + 1;
    $("check-out").innerHTML = c === crop ? `<strong>${cap(c)} is the best match</strong> for these readings (${pct(v)}).`
      : `<strong>${v >= 0.7 ? "Good fit" : v >= 0.3 ? "Marginal" : "Unsuitable"}:</strong> ${cap(c)} scores ${pct(v)} (rank ${rank} of ${m.classes.length}).
        ${v < 0.3 ? "Very low chance of growing here. " : ""}Best match is ${crop} (${pct(conf)}).`;
  };
  // One explanation for the whole page: how much each reading costs the suitability score (never negative: the model is monotone).
  const med = m.suit.profile[crop].med;
  const gain = m.features.map((f, i) => ({ f, g: Math.max(0, suitability(m, crop, x.map((v, j) => (j === i ? med[i] : v))) - conf) }));
  const worst = [...gain].sort((a, b) => b.g - a.g).filter((e) => e.g >= 0.02).slice(0, 2)
    .map((e) => LABEL[e.f].replace(/ \(.*\)/, "").toLowerCase());
  const maxG = Math.max(...gain.map((e) => e.g), 0.01);
  $("why").innerHTML = `
    <h2>Why this score?</h2>
    <p>${worst.length ? `${cap(crop)} scores ${pct(conf)}, held back mainly by your <strong>${worst.join(" and ")}</strong>.`
      : `Every reading is close to typical for ${crop}, so the score is high.`}</p>
    <p class="muted">Each bar shows how much the ${crop} score would rise if that reading were at the typical value for ${crop}. No bar means the reading is fine.</p>
    <div class="split">
      <div>${gain.map((e) => bar(LABEL[e.f], e.g / maxG, e.g >= 0.02 ? "+" + pct(e.g) : "fine")).join("")}</div>
      <div>${radar(x, m.crop_stats[crop])}<p class="muted center">Your farm (line) vs. the usual band for ${crop} (shaded)</p></div>
    </div>`;

  const stats = m.crop_stats[crop];
  const { facts, fired } = forwardChain(deriveFacts(m.features, x, stats), rules.rules);
  const advice = Object.keys(rules.advice).filter((a) => facts.has(a));
  const irrig = backwardChain("Irrigate", deriveFacts(m.features, x, stats), rules.rules);
  $("advice").innerHTML = `
    <h2>Advice for ${crop}</h2>
    <p class="muted">Your readings are compared with the 10th–90th percentile range of ${crop} in the training data.</p>
    ${advice.length ? `<ul>${advice.map((a) => `<li>${rules.advice[a]}</li>`).join("")}</ul>`
      : `<p>Your conditions are all within the usual range for ${crop}. No corrective action suggested.</p>`}
    <details><summary>How the rule engine got there</summary>
      <p>Forward chaining fired: ${fired.length ? fired.join(", ") : "no rules"}.</p>
      <p><strong>Should we irrigate?</strong> ${irrig.ok ? "Yes" : "No"} (backward chaining):</p>
      <ol>${irrig.proof.map((l) => `<li>${l}</li>`).join("")}</ol>
    </details>`;

  const tree = m.classes[treePredict(m, x)], nn = m.classes[argmax(mlpProba(m, x))], rf = m.classes[argmax(p)];
  const M = m.metrics;
  $("compare").innerHTML = `
    <h2>Cross-check: three classifiers</h2>
    <table><thead><tr><th>Model</th><th>Says</th><th>Test accuracy</th><th>Explainable?</th></tr></thead><tbody>
      <tr><td>Decision tree</td><td>${tree}</td><td>${pct(M.decision_tree.accuracy)}</td><td>Yes, readable rules</td></tr>
      <tr><td>Random forest</td><td>${rf}</td><td>${pct(M.random_forest.accuracy)}</td><td>Partly (importances)</td></tr>
      <tr><td>Neural network</td><td>${nn}</td><td>${pct(M.neural_net.accuracy)}</td><td>No, black box</td></tr>
    </tbody></table>
    <p class="muted">${tree === rf && nn === rf ? (rf === crop ? "All three agree with the suitability pick." : `All three classifiers say ${rf}, but the suitability model prefers ${crop}: treat it as a close call between similar crops.`) : "The classifiers disagree, a sign this is a borderline case."}
      Model files total ${(JSON.stringify(m).length / 1024).toFixed(0)} KB, so they run offline on a phone or low-cost device.</p>`;

  renderYield(crop);
  if (animate) {
    animateResults($("results"));
    pulseCrop($("crop-name"));
    if (innerWidth <= 760) $("rec").scrollIntoView({ behavior: reduced ? "auto" : "smooth" });
  }
}

// Radar: each reading scaled to the dataset range; shaded ring = crop's 10th-90th percentile band.
const SHORT = { N: "N", P: "P", K: "K", temperature: "Temp", humidity: "Hum", ph: "pH", rainfall: "Rain" };
function radar(x, stats) {
  const C = 120, R = 82, n = m.features.length;
  const pt = (i, r) => { const a = (i / n) * 2 * Math.PI - Math.PI / 2; return [C + Math.cos(a) * R * r, C + Math.sin(a) * R * r]; };
  const norm = (f, v) => { const [lo, hi] = m.ranges[f]; return Math.max(0, Math.min(1, (v - lo) / (hi - lo))); };
  const path = (rs) => rs.map((r, i) => pt(i, r).map((v) => v.toFixed(1)).join(",")).join(" ");
  const band = (k) => m.features.map((f) => norm(f, stats[f][k]));
  const you = m.features.map((f, i) => norm(f, x[i]));
  return `<svg class="radar" viewBox="0 0 240 240" role="img" aria-label="Radar chart of your readings against the crop's usual range">
    ${[0.25, 0.5, 0.75, 1].map((r) => `<polygon points="${path(m.features.map(() => r))}" class="grid"/>`).join("")}
    ${m.features.map((f, i) => { const [ex, ey] = pt(i, 1), [lx, ly] = pt(i, 1.18);
      return `<line x1="${C}" y1="${C}" x2="${ex.toFixed(1)}" y2="${ey.toFixed(1)}" class="grid"/><text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" text-anchor="middle" dominant-baseline="middle">${SHORT[f]}</text>`; }).join("")}
    <path d="M${path(band(2)).replaceAll(" ", " L")} Z M${path(band(0)).replaceAll(" ", " L")} Z" class="band" fill-rule="evenodd"/>
    <g class="radar-poly"><polygon points="${path(you)}" class="you"/>
    ${you.map((r, i) => `<circle cx="${pt(i, r)[0].toFixed(1)}" cy="${pt(i, r)[1].toFixed(1)}" r="3"/>`).join("")}</g>
  </svg>`;
}

function bar(label, frac, text = pct(frac)) {
  const w = Math.max(0, Math.min(1, frac)) * 100 + "%";
  return `<div class="bar"><span>${label}</span><i style="--w:${w}" data-w="${w}"></i><b>${text}</b></div>`;
}

// ---- yield (separate model: different dataset, annual country averages) ----
function renderYield(crop) {
  const y = m.yield, mapped = y.for_crop[crop];
  const prev = $("yield").querySelector("select")?.value;
  const item = mapped || prev || y.items[0];
  $("yield").innerHTML = `
    <h2>Yield estimate</h2>
    <p class="muted">${mapped ? `Yield data exists for ${crop}.` : `The yield dataset has no data for ${crop}, so pick a crop it does cover.`}
      This model was trained on yearly country averages, so enter <em>annual</em> values.</p>
    <div class="row">
      <label>Crop<select id="y-item">${y.items.map((i) => `<option${i === item ? " selected" : ""}>${i}</option>`).join("")}</select></label>
      <label>Annual rainfall (mm)<input id="y-rain" type="number" value="1000" step="10"></label>
      <label>Pesticides (tonnes)<input id="y-pest" type="number" value="${y.median_pesticides}" step="10"></label>
      <label>Avg temperature (°C)<input id="y-temp" type="number" value="${$("in-temperature").value}" step="0.1"></label>
    </div>
    <p class="big" id="y-out"></p>
    <p class="muted">Linear regression: R² ${y.metrics.R2}, typical error ±${(y.metrics.MAE / 10000).toFixed(1)} t/ha. It sees only 3 weather/chemical inputs
      and the crop type, so use it for a rough idea only.</p>`;
  const upd = () => {
    const v = ["y-rain", "y-pest", "y-temp"].map((id) => parseFloat($(id).value));
    $("y-out").textContent = v.every(Number.isFinite)
      ? `≈ ${predictYield(m, $("y-item").value, ...v).toFixed(1)} tonnes per hectare` : "Enter all values";
  };
  $("yield").querySelectorAll("input,select").forEach((e) => (e.oninput = upd));
  upd();
}

// ---- boot ----
{
  const [crops, acc, kb] = [m.classes.length, m.metrics.random_forest.accuracy * 100, JSON.stringify(m).length / 1024];
  const d = document.querySelectorAll("#hero-stats dd");
  d[0].dataset.count = crops; d[1].dataset.count = acc.toFixed(1); d[3].dataset.count = Math.round(kb);
}
run();
initScroll();
intro();
scrollReveals();
startHero($("hero-canvas"), reduced);
