# Study 1 — simulation

A community with less common knowledge tends to have a higher degree of transparency.

Each meaning can be encoded two ways:

| encoding | production cost | learning cost | works with strangers? |
|---|---|---|---|
| **transparent** (built from parts) | high (long) | none (inferable) | yes |
| **conventional** (short, memorized) | low (short) | one-time, per learner | only if the hearer shares it |

A population minimizes communication and learning cost. Memory amortizes over frequency times the number of people who share the form, so conventionalizing a meaning pays off only when it is both frequent and widely shared.

**Society type** enters through shared exposure `E` — how many community-wide exposures a typical learner accumulates. Less common knowledge (small `E`) leaves more of the lexicon transparent. Close-knit societies (large `E`) can conventionalize further down the frequency tail. The share of vocabulary that is commonly known is an output of `E`, and it caps how far conventionalization can profitably reach.

For meaning *i* with frequency *f_i*, knowledge of its convention is `q_i = 1 - exp(-λ · E · f_i)`. The society conventionalizes *i* when the expected conventional cost beats the transparent cost.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python society_model.py       # numeric summary of three societies
python run_experiments.py     # write figures/
```

Figures, in narrative order:

1. `figures/cost_vs_society.png` — mixed encoding beats both naive strategies. All-transparent is flat. All-conventional is U-shaped.
2. `figures/shared_knowledge.png` — `E` sets how far into the frequency tail a random hearer knows a convention.
3. `figures/encoding_and_lexicon.png` — conventionalize the frequent core; the tail stays transparent. Even open societies conventionalize a few types that carry most of the usage.

Knobs live in `ModelParams` (`society_model.py`): `V`, `zipf_s`, `cost_comp`, `cost_conv`, `cost_repair`, `mem_cost`, `horizon`, `lam`.
