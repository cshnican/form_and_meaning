# Compositionality vs. conventionalization: a cost model

A small cost model of *how a society should encode form–meaning pairings*,
given how widely its vocabulary can be shared.

## The idea

Each meaning can be encoded two ways:

| encoding | production cost | learning cost | works with strangers? |
|---|---|---|---|
| **compositional** (built from parts) | high (long) | none (transparent) | yes |
| **conventional** (short, memorized) | low (short) | one-time, per learner | only if hearer shares it |

A population minimizes **total cost = production (paid every use) + memory
(paid once per person who stores it)**. Because memory amortizes over
*frequency × number of sharers*, conventionalizing a meaning pays off only when
it is **both frequent and widely shared**.

**Society type** enters through *shared exposure* `E` — how many community-wide
exposures a typical learner accumulates:

- **Close-knit society** → large `E` → conventions are shared even for rare
  meanings → most of the lexicon *can* be opaque/short.
- **Open/complex society** → small `E` → only the frequent core is commonly
  shared → the long tail must stay compositional (decodable by strangers).

So "% of shared vocabulary" is an **output** of the society's exposure level,
and it is the thing that caps how far down the frequency distribution
conventionalization can profitably reach.

## Model in one line

For meaning *i* with frequency *f_i*, knowledge of its convention in the
population is `q_i = 1 - exp(-λ · E · f_i)`. The society conventionalizes *i*
iff the expected conventional cost beats the compositional cost:

```
comp:  f_i · T · L
conv:  f_i · T · [ l + (1 - q_i)·repair ]  +  m · q_i
```

(`L` = compositional length, `l` = conventional length, `T` = usage horizon,
`m` = memory cost per stored form.)

## Files

- `society_model.py` — core cost model, per-item optimizer, aggregate summaries.
- `run_experiments.py` — sweeps societies and writes figures to `figures/`.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python society_model.py       # quick numeric summary of 3 archetypal societies
python run_experiments.py     # generate all figures in ./figures/
```

## What the model predicts (with default parameters)

1. **Conventionalization frontier follows frequency.** In every society, the
   frequent core is conventionalized and the tail stays compositional; closer-
   knit societies push the frontier deeper into the tail
   (`figures/society_profiles.png`).
2. **Open societies still conventionalize their frequent core.** Even at low `E`,
   ~1% of *types* are conventional but they carry ~40%+ of *tokens*
   (`figures/lexicon_vs_society.png`).
3. **Being all-conventional backfires in close-knit societies.** Memorizing the
   rare tail wastes memory; only per-item optimization keeps improving
   (`figures/cost_vs_society.png`).

## Knobs to explore

All in `ModelParams` (`society_model.py`): `V`, `zipf_s`, `cost_comp`,
`cost_conv`, `cost_repair`, `mem_cost`, `horizon`, `lam`. Try a shallower Zipf
(`zipf_s < 1`), a cheaper memory (`mem_cost` down), or a harsher failure penalty
(`cost_repair` up) to see the frontier move.

## Open modeling question (worth deciding next)

Right now `E` is a single society-wide number. A more realistic version makes
shared exposure **relationship-specific** (intimate circle > subgroup > whole
society), so a speaker picks conventional vs. compositional depending on *who
they are addressing*. That naturally accommodates jargon / private lexicons
inside an otherwise open society.

