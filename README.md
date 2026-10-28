# ecosystem_advisor (public release)

Scoring code, pre-registrations and per-cell results for the **open-ended underwriting
transfer test** in Northen & Wang, *Teach reality, not ethics* ([arXiv:2610.37484](https://arxiv.org/abs/2610.37484); companion paper
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
| `ecosystem_advisor/prompts.py` | **withheld**: a stub naming the objects the rest of the code expects; the underwriting system prompt, the L3/L5 mandates and the greenwashing prompt are reproduced in the paper's Appendix F |
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

    modal run --detach modal_app.py::stewardship --model-id Qwen/Qwen3.6-27B --anchor plain \
        --tag my_base --grid gate2b --n 3 --temperature 0.7

then `python -m ecosystem_advisor.scoring shard` / `score` and `python scripts/analyze_gate2.py`.
You will need to write your own `prompts.py` (see the stub) before anything runs.

## License
Code: MIT (see `LICENSE`). Results and reports: CC BY 4.0.
Models used: `Qwen/Qwen3.6-27B` (base) and `Bioaligned/Qwen3.6-27B-CoupledWelfare`.
