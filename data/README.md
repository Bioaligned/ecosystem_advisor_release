# Item schema (items themselves withheld)

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
