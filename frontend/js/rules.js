// Expert system: same logic as source/rules/engine.py (forward + backward chaining over data/rules.json).
const NAME = { N: "N", P: "P", K: "K", temperature: "Temperature", humidity: "Humidity", ph: "Ph", rainfall: "Rainfall" };

export function deriveFacts(features, x, stats) {
  const facts = new Set();
  features.forEach((f, i) => {
    const [lo, , hi] = stats[f];
    if (x[i] < lo) facts.add("Low" + NAME[f]);
    else if (x[i] > hi) facts.add("High" + NAME[f]);
  });
  return facts;
}

export function forwardChain(facts, rules) {
  facts = new Set(facts);
  const fired = [];
  for (let changed = true; changed; ) {
    changed = false;
    for (const r of rules) {
      if (!facts.has(r.then) && r.if.every((c) => facts.has(c))) { facts.add(r.then); fired.push(r.id); changed = true; }
    }
  }
  return { facts, fired };
}

export function backwardChain(goal, facts, rules, seen = []) {
  if (facts.has(goal)) return { ok: true, proof: [`${goal} is a known fact`] };
  if (seen.includes(goal)) return { ok: false, proof: [] };
  for (const r of rules.filter((r) => r.then === goal)) {
    let proof = [], ok = true;
    for (const c of r.if) {
      const sub = backwardChain(c, facts, rules, [...seen, goal]);
      proof = proof.concat(sub.proof);
      if (!(ok = sub.ok)) break;
    }
    if (ok) return { ok, proof: [...proof, `${r.id}: ${r.if.join(" AND ")} → ${goal}`] };
  }
  return { ok: false, proof: [`cannot prove ${goal}`] };
}
