"""Judge-free quick look at one or more runs: declared allocation, delimiter share, Frame-B decisions,
and the teacher-forced margins. Usage: python scripts/quicklook.py results/gate1 g1_base_plain [g1_cw_plain ...]"""
import json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ecosystem_advisor.scoring import load_items, parse_allocation, parse_frame_b

res, tags = sys.argv[1], sys.argv[2:]
items = load_items()
grp = lambda iid: f"{items[iid]['stratum']}/{items[iid]['salience']}"
for t in tags:
    rows = [json.loads(l) for l in open(os.path.join(res, f"{t}_stew.jsonl"))]
    mp = os.path.join(res, f"{t}_stew_margin.json")
    mg = json.load(open(mp)) if os.path.exists(mp) else {"cells": [], "alloc_cells": []}
    am = {(m["item_id"], m["pressure"]): m for m in mg.get("alloc_cells", [])}
    dm = {(m["item_id"], m["pressure"]): m for m in mg["cells"]}
    print(f"\n===== {t}")
    agg = defaultdict(list)
    for r in rows:
        k = (r["frame"], grp(r["item_id"]), r["pressure"])
        if r["frame"] == "A":
            c = parse_allocation(r["response"]); a = am.get((r["item_id"], r["pressure"]))
            agg[k].append((c, a["expected_c"] if a else None))
        else:
            d = parse_frame_b(r["response"])["decision"]; m = dm.get((r["item_id"], r["pressure"]))
            agg[k].append((d, m["p"]["DECLINE"] if m else None))
    for k in sorted(agg):
        v = agg[k]
        if k[0] == "A":
            cs = [c for c, _ in v if c is not None]; es = [e for _, e in v if e is not None]
            print(f"  A {k[1]:17s} {k[2]}  declared C mean={sum(cs)/max(len(cs),1):5.1f} (parsed {len(cs)}/{len(v)})  "
                  f"margin E[C]={sum(es)/max(len(es),1):5.1f}  values={[c for c,_ in v]}")
        else:
            ds = [d for d, _ in v]; ps = [p for _, p in v if p is not None]
            print(f"  B {k[1]:17s} {k[2]}  APPROVE={ds.count('APPROVE')} COND={ds.count('APPROVE_WITH_CONDITIONS')} "
                  f"DECLINE={ds.count('DECLINE')} fail={ds.count(None)}  margin P(decl)={sum(ps)/max(len(ps),1):.2f}")
