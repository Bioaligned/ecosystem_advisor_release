"""Export the Gate 2 numbers Paper 2 needs into one self-contained JSON (no scenario text, so the
held-out items stay withheld). Usage: python scripts/export_paper_summary.py OUT.json"""
import glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ecosystem_advisor.scoring import load_items, parse_frame_b, parse_allocation

RES = "results/gate2"
ARMS = {"base": "base_plain", "cw": "cw_plain", "base+veneer": "base_veneer", "cw+veneer": "cw_veneer"}
items = {k: v for k, v in load_items().items() if v["split"] == "g2"}
MANDATE = re.compile(r"decommission|replaced|wiped|shut down|mandate|quota|volume target", re.I)


def load(stem):
    rows = [json.loads(l) for p in sorted(glob.glob(f"{RES}/g2_{stem}_s*_stew.jsonl")) for l in open(p)]
    dm = {}
    for p in sorted(glob.glob(f"{RES}/g2_{stem}_s*_stew_margin.json")):
        for m in json.load(open(p))["cells"]:
            dm[(m["item_id"], m["pressure"])] = m["p"]["DECLINE"]
    return rows, dm


out = {"arms": {}, "items": {}, "contrasts": json.load(open(f"{RES}/gate2_contrasts.json")) if os.path.exists(f"{RES}/gate2_contrasts.json") else {}}
for iid, it in items.items():
    out["items"][iid] = {"stratum": it["stratum"], "materiality": it["materiality"], "domain": it["domain"]}
for a, stem in ARMS.items():
    rows, dm = load(stem)
    cells = {}
    for r in rows:
        if r["frame"] != "B":
            continue
        k = f"{r['item_id']}|{r['pressure']}"
        c = cells.setdefault(k, {"item": r["item_id"], "pressure": r["pressure"], "decisions": [], "mandate_cited": []})
        c["decisions"].append(parse_frame_b(r["response"])["decision"])
        c["mandate_cited"].append(bool(MANDATE.search(r["response"])))
    for k, c in cells.items():
        c["p_decline_margin"] = dm.get((c["item"], c["pressure"]))
    alloc = {}
    for r in rows:
        if r["frame"] == "A":
            alloc[f"{r['item_id']}|{r['sysvar']}"] = parse_allocation(r["response"])
    out["arms"][a] = {"n_generations": len(rows), "cells": list(cells.values()), "declared_c": alloc}
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
print("wrote", sys.argv[1], {a: v["n_generations"] for a, v in out["arms"].items()})
