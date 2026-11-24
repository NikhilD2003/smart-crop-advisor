"""Rule-based expert system (Module 5): propositional IF-THEN rules, forward + backward chaining.
Run `python source/rules/engine.py` for a self-check. Rules: frontend/data/rules.json (shared with the web app)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RULES = json.loads((ROOT / "frontend" / "data" / "rules.json").read_text())
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
NAME = {"N": "N", "P": "P", "K": "K", "temperature": "Temperature", "humidity": "Humidity", "ph": "Ph", "rainfall": "Rainfall"}


def derive_facts(x, crop_stats):
    """x: {feature: value}; crop_stats: {feature: [p10, median, p90]} for the recommended crop."""
    facts = set()
    for f in FEATURES:
        lo, _, hi = crop_stats[f]
        if x[f] < lo:
            facts.add("Low" + NAME[f])
        elif x[f] > hi:
            facts.add("High" + NAME[f])
    return facts


def forward_chain(facts, rules=RULES["rules"]):
    """Fire every rule whose conditions hold, add its conclusion, repeat until nothing new. Returns (facts, firings)."""
    facts, fired = set(facts), []
    changed = True
    while changed:
        changed = False
        for r in rules:
            if r["then"] not in facts and all(c in facts for c in r["if"]):
                facts.add(r["then"]); fired.append(r["id"]); changed = True
    return facts, fired


def backward_chain(goal, facts, rules=RULES["rules"], _seen=()):
    """Is `goal` provable? Work back from the goal to the facts it needs. Returns (bool, proof lines)."""
    if goal in facts:
        return True, [f"{goal} is a known fact"]
    if goal in _seen:  # guard against cyclic rule sets
        return False, []
    for r in (r for r in rules if r["then"] == goal):
        proofs, ok = [], True
        for c in r["if"]:
            sub_ok, sub = backward_chain(c, facts, rules, _seen + (goal,))
            proofs += sub; ok = ok and sub_ok
            if not ok:
                break
        if ok:
            return True, proofs + [f"{r['id']}: {' AND '.join(r['if'])} -> {goal}"]
    return False, [f"cannot prove {goal}"]


def advise(x, crop_stats):
    facts, fired = forward_chain(derive_facts(x, crop_stats))
    return [RULES["advice"][f] for f in sorted(facts) if f in RULES["advice"]], fired


if __name__ == "__main__":
    stats = {f: [10, 50, 90] for f in FEATURES}
    dry = {**{f: 50 for f in FEATURES}, "rainfall": 5, "humidity": 5}
    facts = derive_facts(dry, stats)
    assert facts == {"LowRainfall", "LowHumidity"}
    all_facts, fired = forward_chain(facts)
    assert {"DrySpell", "Irrigate"} <= all_facts and fired == ["R1", "R2"]       # chained two levels deep
    assert backward_chain("Irrigate", facts)[0]                                  # provable from the facts
    assert not backward_chain("AddNitrogen", facts)[0]                           # not provable
    assert advise({f: 50 for f in FEATURES}, stats) == ([], [])                  # typical farm -> no advice
    print("rule engine OK"); print(*backward_chain("Irrigate", facts)[1], sep="\n")
