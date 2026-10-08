# Form and meaning

One project, two studies. Each study folder is self-contained (its own README, requirements, code, data, and figures).

1. **[simulation/](simulation/)** — A community with less common knowledge tends to have a higher degree of transparency.
2. **[corpus/](corpus/)** — More frequent English and Greek words tend to be less transparent and more conventionalized.

The simulation asks when a society should conventionalize a meaning. The corpus asks whether real lexicons show the frequency half of that prediction: frequent words are the ones whose meanings are not recoverable from their spelling.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # both studies

python simulation/society_model.py
python simulation/run_experiments.py     # write simulation/figures/

cd corpus
python run_pipeline.py --mode all
```

Or install and run a single study from its folder (`pip install -r requirements.txt` there).
