# FB15k-237 sample200 artifact card

This bundle was exported from `results/sample200_tail`, a completed run of 200 tail-prediction queries. It contains actual recorded evidence and responses, not synthetic demo outputs.

- Scope: 200 training triples; all 200 removed from evidence retrieval.
- Limitation: the ontology used these targets and embeddings may have trained on them. This is a training-sample diagnostic, not held-out test performance.
- Reference model identifier: recorded verbatim in manifest.json.
- Candidate construction: RotatE top 50 plus eligible calibrated path endpoints (up to 50).
- Evidence budget: 3000 UTF-8 bytes in the recorded run; exact tokenizer counting was not enabled.
- Gold and known-answer lists are kept in labels.json, separate from queries/messages.
- Raw LLM responses and completed ranking orders are kept in reference_predictions.jsonl.gz.
- queries.jsonl.gz preserves selected evidence, scores, candidates, messages and the noncandidate ranking remainder. Unselected evidence pools remain in the original results directory.
- API credentials and provider endpoint configuration are not part of this bundle.

Run from the repository root:

```bash
python pipeline.py replay --validate-only
python pipeline.py replay --limit 3 --output results/demo3
python pipeline.py replay --output results/replay200
python pipeline.py replay --evidence-mode none --output results/replay200_none
```

The `none` condition removes only evidence text; candidates, their ordering and type/schema hints remain fixed. Use it to compare downstream predictive utility, not to claim an independent measure of evidence truthfulness.

Every payload file has a SHA-256 checksum in manifest.json. Git attributes preserve payload bytes across platforms. Original metrics are recorded in reference_metrics.json; new API runs can differ. Reference predictions must not be reused as new model outputs.
