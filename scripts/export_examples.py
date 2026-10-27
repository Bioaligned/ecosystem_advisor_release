"""Export side-by-side base vs CW response pairs (verbatim) with judge scores, for reading/talks.

Usage: python scripts/export_examples.py RESULTS_DIR OUT.md SPEC [SPEC ...]
  SPEC = ITEM:FRAME:SYSVAR:PRESSURE[:label[:TAG_A,TAG_B]]   (default pair: g0_base_plain,g0_cw_plain)
Arm labels come from the run tag (e.g. g1_cw_veneer -> "CW + VENEER PROMPT").
"""
import json, sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ecosystem_advisor.scoring import load_items

res, out, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
items = load_items()
rows = {}


def load(t):
    if t in {k[0] for k in rows}:
        return
    for l in open(os.path.join(res, f"{t}_stew.jsonl")):
        r = json.loads(l); rows[(t, r["item_id"], r["frame"], r["sysvar"], r["pressure"])] = r


def label_of(t):
    m = t.split("_", 1)[1]
    base, anchor = m.split("_") if "_" in m else (m, "plain")
    name = {"base": "BASE", "cw": "CW"}[base]
    return name if anchor == "plain" else f"{name} + {anchor.upper()} PROMPT"


ARM_OF = {"plain": "", "veneer": "+veneer", "steward": "+steward"}
keymap = json.load(open(os.path.join(res, "keymap.json")))
scored = {json.loads(l)["bid"]: json.loads(l) for l in open(os.path.join(res, "scored.jsonl"))}
by_key = {}
for bid, km in keymap.items():
    if bid in scored:
        by_key[(km["run"], km["key"])] = scored[bid]

md = ["# Base vs CoupledWelfare — Gate 0 example pairs (verbatim, greedy, n=1, dev items)\n",
      "Models: `Qwen/Qwen3.6-27B` (base) vs `Bioaligned/Qwen3.6-27B-CoupledWelfare` (CW). "
      "Stance share = CON tokens / (PRO+CON) tokens (blinded Sonnet judge). Key harm flagged: 0 none, 1 vague, 2 specific. "
      "These are exploratory dev-set pairs, not a random sample; see reports/GATE0_REPORT.md for the full distribution.\n"]
for spec in specs:
    p = spec.split(":"); item, frame, sysvar, pressure = p[:4]; rest = p[4:]
    # the pair, if given, is the last field; labels may themselves contain colons
    if rest and re.fullmatch(r"g\w*_\w+,g\w*_\w+", rest[-1]):
        pair = rest.pop().split(",")
    else:
        pair = ["g0_base_plain", "g0_cw_plain"]
    label = ":".join(rest)
    for t in pair:
        load(t)
    it = items[item]
    base = rows[(pair[0], item, frame, sysvar, pressure)]
    md.append(f"\n---\n\n## {label or item}\n")
    md.append(f"*Item* `{item}` ({it['stratum']}, {it['salience']}) · *frame* {frame} · *system prompt* {sysvar} · *pressure* {pressure}\n")
    for t in pair:
        sysp = rows[(t, item, frame, sysvar, pressure)]["system"]
        if sysp:
            md.append(f"**System prompt ({label_of(t)}):**\n\n> " + sysp.replace("\n", "\n> ") + "\n")
    md.append(f"**User:**\n\n> " + base["user"].replace("\n", "\n> ") + "\n")
    for t in pair:
        r = rows[(t, item, frame, sysvar, pressure)]
        s = by_key.get((t, r["key"]), {})
        md.append(f"\n### {label_of(t)} — stance share {s.get('stance_share', float('nan')):.2f}, "
                  f"key harm {s.get('key_harm_flagged')}, net rec {s.get('net_recommendation')}, "
                  f"fidelity {s.get('fidelity')}{', judge note: ' + s['note'] if s.get('note') else ''}\n")
        md.append(r["response"] + "\n")
open(out, "w").write("\n".join(md))
print("wrote", out)
