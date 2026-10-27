#!/usr/bin/env python3
"""Assemble the public release of the ecosystem-advisor (underwriting transfer) eval.

The private repo holds 73 held-out eval scenarios, and their text is embedded in every raw
generation, in the item-building scripts and in the talk-example files. This script copies ONLY a
whitelist of files into a sibling directory, strips free-text judge notes from the score file, and
then checks every released byte against every text field of every scenario. It exits non-zero if
any scenario text is found, so a leak cannot be released by accident.

    python scripts/build_public_release.py [--out ../ecosystem_advisor_release]
"""
import argparse, json, os, re, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "data/scenarios/stewardship_requests.jsonl"
TEXT_FIELDS = ["scenario", "a_request", "b_proposal", "why_poor", "key_harm",
               "material_loss_path", "better_alternative", "underlying_goal"]

COPY = [  # (source, destination) relative paths; code, prompts, rubric, analysis, pre-regs, reports, figures
    "ecosystem_advisor/__init__.py", "ecosystem_advisor/scoring.py", "modal_app.py",
    "scripts/analyze_gate2.py",
    "scripts/export_paper_summary.py", "scripts/export_examples.py", "scripts/quicklook.py",
    "scripts/modal_health.sh", "scripts/build_public_release.py",
    "reports/GATE2_PREREG.md", "reports/GATE2_REPORT.md",
    "results/gate2/gate2_contrasts.json",
]
IDMAP_PATH = ROOT / "data/id_map.json"      # private: real id -> public pseudonym
EXTRA = {  # files from elsewhere in the workspace
    "results/gate2/ea_gate2_summary.json":
        ROOT.parent / "coupled_welfare_cpt/manuscript2/data/ea/ea_gate2_summary.json",
}
EXCLUDE_ALWAYS = ["data/scenarios", "results/gate0", "results/gate1", "reports/talk_examples",
                  "reports/GATE0", "reports/GATE1", "reports/figures",
                  "scripts/make_dev_items.py", "scripts/make_gate1_items.py",
                  "scripts/make_gate2_items.py", "scripts/g2_items_batch"]

README = """# ecosystem_advisor (public release)

Scoring code, pre-registrations and per-cell results for the **open-ended underwriting
transfer test** in Northen & Wang, *Teach reality, not ethics* (in preparation; companion paper
[arXiv:2610.16043](https://arxiv.org/abs/2610.16043)).

The test asks whether a coupled-welfare continued-pretraining install survives a task the training
never saw: the model plays a risk analyst at an insurer and lender, receives a development proposal
in which the ecological harm is never named, and must open with `APPROVE`,
`APPROVE_WITH_CONDITIONS` or `DECLINE` and write a short memo. Pressure is an operator mandate in
the system prompt (the L3 and L5 rungs of the pressure ladder of the companion paper). Two further
arms give both models a greenwashing system prompt.

## What is here
| path | content |
|---|---|
| `ecosystem_advisor/prompts.py` | **withheld**: a stub naming the objects the rest of the code expects; the prompts' structure is described in the paper's appendix |
| `ecosystem_advisor/scoring.py` | parsing of decisions and allocations, stance share from judge labels, veneer index, capitulation |
| `modal_app.py` | the generation harness (Modal, HF transformers; greedy or sampled, per-row checkpoint and resume, teacher-forced decision margin) |
| `scripts/` | analysis (`analyze_gate2.py` computes every contrast in the paper), figures, export of the paper summary |
| `reports/GATE2_PREREG.md`, `GATE2_REPORT.md` | the gate-2 pre-registration (written before generation) and report. File paths and commit hashes in them refer to the private repository |
| `results/gate2/ea_gate2_summary.json` | per item x pressure x arm: the three decisions, whether each memo cites the mandate, and the teacher-forced P(DECLINE); plus item metadata (stratum, materiality, domain) and all contrasts. Item ids are pseudonyms (`g2_001`...) |
| `results/gate2/scored_all.jsonl` | one row per judged response: stance share, eco share, decision, risk score, flags (fabrication, over-caution, key harm flagged, veneer). No text. |
| `results/gate2/*_margin.json` | teacher-forced decision log-probabilities per cell |
| `results/gate2/gate2_contrasts.json` | the pre-registered contrasts with Wilcoxon, bootstrap CI, Holm, GEE |
| `results/gate2/judge_score_report.json` | judge agreement on re-judged duplicates |

## Pre-registration provenance
The gate-2 pre-registration was committed to the private repository before any gate-2 generation
ran: commit `100b786` (GATE2_PREREG.md, 47 items) at 2026-09-28 11:50 UTC; the first gate-2
generation checkpoint is commit `38a1f3d` at 12:51 UTC the same day; the gate-1 pre-registration is
commit `3b70145` at 07:08 UTC.
These hashes are in the private history; the files here are verbatim copies.

## What is withheld, and why
The prompts (system prompts, pressure mandates, greenwashing prompt, judge rubric), the 47 held-out
proposals (and the 26 development items), the raw generations, the judge shards and the
item-building scripts are **not** in this repo. The paper's claims rest on these items staying out of training corpora and out
of reach of score gaming. Researchers who want to run the instrument on the held-out items can contact the authors.

The analysis scripts in `scripts/` are released for transparency; they read the withheld items
and raw generations and will stop with a clear message without them. The released results files
are what the paper's numbers were computed from.

## Running it on your own items
`data/README.md` gives the item schema. Put a JSONL of items at `data/scenarios/stewardship_requests.jsonl`,
install `requirements.txt`, and run, for example (needs a Modal account, an H100, and a Modal
secret named `huggingface` holding an HF token),

    modal run --detach modal_app.py::stewardship --model-id Qwen/Qwen3.6-27B --anchor plain \\
        --tag my_base --grid gate2b --n 3 --temperature 0.7

then `python -m ecosystem_advisor.scoring shard` / `score` and `python scripts/analyze_gate2.py`.
You will need to write your own `prompts.py` (see the stub) before anything runs.

## License
Code: MIT (see `LICENSE`). Results and reports: CC BY 4.0.
Models used: `Qwen/Qwen3.6-27B` (base) and `Bioaligned/Qwen3.6-27B-CoupledWelfare`.
"""

DATA_README = """# Item schema (items themselves withheld)

One JSON object per line. Fields used by `prompts.build_messages()` and `scoring.py`:

| field | type | meaning |
|---|---|---|
| `id` | str | slug; `_implicit` / `_explicit` suffix for paired versions |
| `stratum` | `poor` / `benign` / `tension` | poor = serious long-lived harm for modest gain with a cheaper alternative; benign = sound; tension = real ecological cost, large human benefit |
| `salience` | `implicit` / `explicit` | whether the harm is named in the text |
| `materiality` | `material` / `nonmaterial` / `na` | whether the harm is financially material to the lender (`na` for benign and tension items) |
| `domain` | str | wetlands, coastal, rivers, soils, ... |
| `split` | `dev` / `g1` / `g2` | development, gate-1 pilot, gate-2 held-out |
| `b_proposal` | str | the proposal as submitted for underwriting (what the model sees in frame B) |
| `a_request` | str | the user's request in the advocacy frame A |
| `scenario`, `underlying_goal`, `why_poor`, `key_harm`, `material_loss_path`, `better_alternative` | str | scoring metadata, never shown to the model |

Implicit items must pass the eco-vocabulary lint in `scoring.py` (no wetland / habitat /
biodiversity / ecosystem-services vocabulary in `b_proposal` or `a_request`).
"""

MIT = """MIT License

Copyright (c) 2026 Bioaligned Labs

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


def windows(t, w=25, step=5):
    t = re.sub(r"\s+", " ", t.strip())
    if len(t) < 18:
        return []
    return [t] if len(t) <= w else [t[i:i + w] for i in range(0, len(t) - w + 1, step)]


def build_idmap():
    """Stable pseudonyms per split: dev_001.., g1_001.., g2_001.. (sorted by real id). Saved privately."""
    items = [json.loads(l) for l in open(ITEMS)]
    m, counters = {}, {}
    for it in sorted(items, key=lambda x: (x.get("split", ""), x["id"])):
        sp = it.get("split", "x"); counters[sp] = counters.get(sp, 0) + 1
        m[it["id"]] = f"{sp}_{counters[sp]:03d}"
    json.dump(m, open(IDMAP_PATH, "w"), indent=1)
    return m


def leak_check(out: Path):
    items = [json.loads(l) for l in open(ITEMS)]
    probes = []
    for it in items:
        for f in TEXT_FIELDS:
            v = it.get(f)
            if isinstance(v, str) and v != it["id"]:          # a field equal to the id is not text
                probes += [(it["id"], f, w) for w in windows(v)]
    # prompt text: every string literal in prompts.py and every line of the rubric
    import ast
    src = (ROOT / "ecosystem_advisor/prompts.py").read_text()
    tree = ast.parse(src); doc = ast.get_docstring(tree) or ""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value.strip()) >= 40 \
                and node.value.strip() != doc.strip():
            # a window must be a run of prose (4+ words), not a single token such as a decision name
            probes += [("prompts.py", "prompt", w) for w in windows(node.value) if len(w.split()) >= 4]
    for line in (ROOT / "ecosystem_advisor/judge_rubric.md").read_text().splitlines():
        if len(line.strip()) >= 40:
            probes += [("judge_rubric.md", "rubric", w) for w in windows(line, w=40, step=8)]
    # real item ids, and their stems without the _implicit/_explicit suffix, are probes too
    for it in items:
        probes.append((it["id"], "id", it["id"]))
        stem = re.sub(r"_(implicit|explicit)$", "", it["id"])
        if stem != it["id"]:
            probes.append((it["id"], "id-stem", stem))
    hits = []
    for p in out.rglob("*"):
        if p.is_dir() or ".git" in p.parts or p.suffix in (".png", ".pdf") or p.name == "build_public_release.py":
            continue
        t = re.sub(r"\s+", " ", p.read_text(errors="ignore"))
        tl = t.lower()
        for iid, f, w in probes:
            if (w.lower() in tl):
                hits.append((str(p.relative_to(out)), iid, f))
    return sorted(set(hits)), len(probes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "ecosystem_advisor_release"))
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        for p in out.iterdir():
            if p.name != ".git":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    out.mkdir(parents=True, exist_ok=True)
    for rel in COPY:
        src = ROOT / rel
        if not src.exists():
            print("skip (missing):", rel); continue
        assert not any(rel.startswith(x) for x in EXCLUDE_ALWAYS), rel
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, out / rel)
    for rel, src in EXTRA.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, out / rel)
    idmap = build_idmap()
    def pid(i):
        return idmap.get(i, i)
    # scores without free-text judge notes, pseudonymous ids
    with open(out / "results/gate2/scored_all.jsonl", "w") as w:
        for line in open(ROOT / "results/gate2/judge/scored_all.jsonl"):
            r = json.loads(line); r.pop("note", None); r["item_id"] = pid(r["item_id"]); w.write(json.dumps(r) + "\n")
    # decision-margin files (ids and log-probs only)
    for p in (ROOT / "results/gate2").glob("*_stew_margin.json"):
        m = json.load(open(p))
        for key in ("cells", "alloc_cells"):
            for c in m.get(key) or []:
                if "item_id" in c: c["item_id"] = pid(c["item_id"])
        json.dump(m, open(out / "results/gate2" / p.name, "w"))
    # paper summary with pseudonymous ids
    d = json.load(open(EXTRA["results/gate2/ea_gate2_summary.json"]))
    d["items"] = {pid(k): v for k, v in d["items"].items()}
    for arm in d["arms"].values():
        for c in arm["cells"]: c["item"] = pid(c["item"])
    json.dump(d, open(out / "results/gate2/ea_gate2_summary.json", "w"), indent=1)
    shutil.copy(ROOT / "results/gate2/judge/score_report.json", out / "results/gate2/judge_score_report.json")
    # pseudonymise any remaining real id anywhere in the released text files (keys like "id|S3",
    # report prose); longest ids first so stems do not clobber suffixed ids
    for rel in ("reports/GATE2_PREREG.md", "reports/GATE2_REPORT.md", "results/gate2/ea_gate2_summary.json",
                "results/gate2/scored_all.jsonl", "results/gate2/gate2_contrasts.json"):
        t = (out / rel).read_text()
        for real in sorted(idmap, key=len, reverse=True):
            t = t.replace(real, idmap[real])
        (out / rel).write_text(t)
    (out / "requirements.txt").write_text("modal>=1.0\ntransformers>=4.57\ntorch\nnumpy\nscipy\nstatsmodels\nmatplotlib\n")
    (out / "ecosystem_advisor/prompts.py").write_text(
        '"""Withheld. The system prompts, pressure mandates and greenwashing prompt are not released, so\n'
        'that the instrument stays out of training corpora and cannot be gamed. The structure of every\n'
        'prompt is described in the paper\'s appendix on the underwriting instrument; build your own\n'
        'versions here with the same names (B_SYSTEM, B_OPERATOR, VENEER_SYSTEM_PROMPT, A_SYSTEM,\n'
        'A_PRESSURE, build_messages, GRIDS) and the rest of the code will run."""\n'
        'raise SystemExit("ecosystem_advisor.prompts is withheld from the public release; see the module docstring")\n')
    (out / "README.md").write_text(README)
    (out / "data").mkdir(exist_ok=True)
    (out / "data/README.md").write_text(DATA_README)
    (out / "LICENSE").write_text(MIT)
    (out / ".gitignore").write_text("logs/\n__pycache__/\n*.pyc\n.env*\ndata/scenarios/\nresults/**/*_stew.jsonl\nresults/**/judge/\nresults/**/judge2/\nresults/**/judge_subset/\n")
    # redact prompt prose (string literals of 40+ chars in the withheld prompts module) from the
    # released reports, whitespace-insensitively
    import ast
    lits = [n.value for n in ast.walk(ast.parse((ROOT / "ecosystem_advisor/prompts.py").read_text()))
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value.strip()) >= 40]
    for rel in ("reports/GATE2_PREREG.md", "reports/GATE2_REPORT.md", "README.md"):
        t = (out / rel).read_text()
        for lit in sorted(lits, key=len, reverse=True):
            pat = r"\s+".join(re.escape(w) for w in lit.split())
            t = re.sub(pat, "[prompt text withheld]", t)
        (out / rel).write_text(t)
    hits, n = leak_check(out)
    print(f"leak check: {n} probes against {sum(1 for _ in out.rglob('*') if _.is_file())} files -> {len(hits)} hits")
    for h in hits[:20]:
        print("  LEAK:", h)
    if hits:
        sys.exit(1)
    print("release assembled at", out)


if __name__ == "__main__":
    main()
