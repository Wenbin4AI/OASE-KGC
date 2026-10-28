# Contributing and reporting experiments

Use the supported `pipeline.py` stages; preserve historical scripts for comparison rather than silently changing their behavior.

Before submitting a change:

```bash
pip install -r requirements-quickstart.txt
python -m unittest discover -s tests -v
python pipeline.py replay --validate-only
```

For a research result, record dataset/split, train-only embedding provenance, ontology source, random seed, candidate sizes, evidence budget/tokenizer, model identifier, and raw versus filtered metrics. Include every query, including candidate-recall misses. Report fallback counts and partial runs. Compare evidence ablations with fixed candidates and fixed model settings.

Never place gold answers in ranking prompts or use them to score evidence. Training-sample diagnostics must be labeled separately from held-out results. Do not claim that an API rerun is bitwise deterministic.

Keep API keys in environment variables. Do not commit local API configuration or new experimental output directories. Export intended release artifacts using `pipeline.py export`; validate checksums before sharing. Preserve the original source run and document any changes to released bundles. The archive of the original environment is requirements-legacy.txt; use the minimal supported requirements for new work.
