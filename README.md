# Form and meaning

One study, two parts.

1. **Simulation.** A community with less common knowledge tends to have a higher degree of transparency.
2. **Corpus.** More frequent English and Greek words tend to be less transparent and more conventionalized.

The simulation asks when a society should conventionalize a meaning. The corpus asks whether real lexicons show the frequency half of that prediction: frequent words are the ones whose meanings are not recoverable from their spelling.

## Part 1 — simulation

Code: `society_model.py`, `run_experiments.py`. Figures: `figures/`.

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

## Part 2 — corpus

Code, figures, and a longer write-up: `corpus/` (`corpus/README.md`).

Per-word **transparency** is how well a word's meaning can be predicted from its spelling, relative to a form-blind baseline. The pipeline never builds morpheme vectors. A single map from character n-grams to a GloVe vector is trained on other words, and the target is held out.

```
transparency(w) = cos(g(form_w), v_w) − cos(mean(v_¬w), v_w)
```

Leading principal components of the meaning space are dropped before scoring (all-but-the-top; Mu & Viswanath 2018). The regressions are `transparency ~ zipf_freq` and the same model with a hubness covariate (mean cosine to five nearest neighbors). A negative frequency slope means more frequent words are less transparent.

```bash
cd corpus
python run_pipeline.py --mode morpholex   # MorphoLex ∩ English GloVe 50d
python run_pipeline.py --mode subtlex     # SUBTLEX-US, 30k subsample
python run_pipeline.py --mode subtlex-gr  # SUBTLEX-GR ∩ Greek GloVe 300d
python run_pipeline.py --mode ladec       # LADEC compounds, native Zipf only
python run_pipeline.py --mode all
python score_word.py dog --lexicon data/morpholex_words.csv
```

English results are in `corpus/figures/transparency_vs_freq_english.png` (LADEC, MorphoLex, SUBTLEX-US). Greek results are in `corpus/figures/transparency_vs_freq_greek.png`. Score tables and regression coefficients are written to `corpus/outputs/` on a run.

MorphoLex uses all alphabetic types (not nouns only), wordfreq Zipf, and a random subsample of 30,000 after the GloVe intersect. SUBTLEX-US Zipf is `log10(SUBTLWF)+3`. SUBTLEX-GR Zipf is `log10(SUBTLEX_WF)+3`. LADEC keeps only compounds that have LADEC's own SUBTLEX `Zipfvalue`.
