// Checks the JS models reproduce the Python predictions. Run: node source/tests/parity.mjs
import { readFileSync } from "node:fs";
import { forestProba, treePredict, mlpProba, zone, suitability } from "../../frontend/js/model.js";

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url)));
const m = read("../../frontend/data/model.json"), c = read("./parity_cases.json");
const argmax = (a) => a.reduce((b, v, i) => (v > a[b] ? i : b), 0);

const got = {
  tree: c.x.map((x) => m.classes[treePredict(m, x)]),
  forest: c.x.map((x) => m.classes[argmax(forestProba(m, x))]),
  mlp: c.x.map((x) => m.classes[argmax(mlpProba(m, x))]),
  zone: c.x.map((x) => zone(m, x).id),
};
let bad = false;
for (const k of Object.keys(got)) {
  const match = got[k].filter((v, i) => v === c[k][i]).length / c.x.length;
  console.log(k.padEnd(7), (match * 100).toFixed(1) + "% match");
  if (match < 0.99) bad = true;
}
const maxDiff = Math.max(...c.suit.map(([crop, x, want]) => Math.abs(suitability(m, crop, x) - want)));
console.log("suit    max abs diff", maxDiff.toFixed(4));
if (maxDiff > 0.002) bad = true;
// monotone: pushing any reading further from the crop's typical value must never raise the score
let rnd = 7; const rand = () => (rnd = (rnd * 16807) % 2147483647) / 2147483647;
let viol = 0, checks = 0;
for (let n = 0; n < 400; n++) {
  const crop = m.classes[Math.floor(rand() * m.classes.length)], { med, scale } = m.suit.profile[crop];
  const x = m.features.map((f, i) => med[i] + (rand() * 8 - 4) * scale[i]), i = Math.floor(rand() * 7);
  const y = [...x]; y[i] = med[i] + (x[i] - med[i]) * 1.5; // same side, further out
  checks++; if (suitability(m, crop, y) > suitability(m, crop, x) + 1e-9) viol++;
}
console.log(`monotone ${checks - viol}/${checks} ok`);
if (viol) bad = true;
if (bad) { console.error("PARITY FAILED"); process.exit(1); }
console.log("parity OK");
