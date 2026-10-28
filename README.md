# OASE-KGC

**Ontology-Aware Subgraph Evidence Optimization for Knowledge Graph Completion**

## 1. Framework

![OASE-KGC framework](assets/framework.png)

OASE-KGC provides a frozen LLM with compact, query-specific evidence for knowledge graph completion. RotatE retrieves relevant one-hop neighbors and multi-hop paths. An ontology containing entity types, class relations, and relation domain/range constraints calibrates their semantic compatibility. Budgeted selection then balances evidence quality and structural coverage before the LLM ranks candidate entities.

## 2. Quick start

Use the included 200-query evidence bundle to try KGC without training RotatE or extracting an ontology. Python 3.10+ and an API supporting Chat Completions are required. Run commands from the repository root.

```bash
pip install -r requirements-quickstart.txt
cp ontology_config.example.json ontology_config.local.json
```

Set `base_url` and `model` in the local configuration. Keep `api_key_env` as `LLM_API_KEY` and set your key:

```bash
export LLM_API_KEY='YOUR_API_KEY'
python pipeline.py replay --limit 3 --output results/quickstart
```

Remove `--limit 3` and use a new output directory to run all 200 queries. Results include predictions, LLM responses, evidence, and evaluation metrics. The included bundle is a training-sample demonstration; see its [artifact card](artifacts/fb15k237-sample200/README.md) for provenance.

## 3. Run from scratch

### 3.1 Environment

```bash
conda create -n oase-kgc python=3.10 -y
conda activate oase-kgc
pip install -r requirements.txt
```

Configure the LLM API as described in Quick start. The following examples use FB15k-237; its files are provided under `dataset/FB15k-237/`.

### 3.2 Extract the ontology with an LLM

```bash
python pipeline.py ontology \
  --data-path dataset/FB15k-237 \
  --config ontology_config.local.json --mode all
```

This constructs entity types, subclass/disjoint relations, and relation domain/range constraints from the training graph. The result is saved to `import/ontology/FB15k-237/all/ontology.json`. Use `--mode sample` to try 200 training triples instead.

### 3.3 Extract ontology-aware evidence

First train RotatE, or reuse compatible training-only weights:

```bash
python pipeline.py train --mode train \
  --data_path dataset/FB15k-237 --dataset_name FB15k-237 \
  --hidden_dim 200 --gamma 9 --learning_rate 0.00005 --max_steps 100000
```

Then retrieve, calibrate, and select evidence:

```bash
python pipeline.py evidence \
  --data-path dataset/FB15k-237 \
  --ontology import/ontology/FB15k-237/all/ontology.json \
  --embeddings import/KGE_model/FB15k-237 \
  --split test --prediction tail --limit 10 \
  --encoding o200k_base --token-budget 3000 \
  --output output/evidence/test10.jsonl
```

Each output record contains `evidence_text`, selected evidence with scores, candidate entities, and `llm_messages` ready for an LLM. Remove `--limit 10` to extract evidence for the complete split.

### 3.4 Perform KGC with the evidence

The evaluator combines the same evidence-construction procedure with LLM ranking and metric computation in one command; it does not read the inspection JSONL from the previous step:

```bash
python pipeline.py evaluate \
  --data-path dataset/FB15k-237 \
  --ontology import/ontology/FB15k-237/all/ontology.json \
  --embeddings import/KGE_model/FB15k-237 \
  --split test --sample-size 200 --seed 2026 --prediction both \
  --encoding o200k_base --token-budget 3000 \
  --output results/test200
```

Remove `--sample-size 200` to evaluate the complete test split. Use `--prediction tail` for tail prediction only. Outputs include `metrics.json`, correct-first prediction reports, raw LLM responses, and per-query evidence. Rerun the same command to resume an interrupted evaluation.

To preserve this experiment's evidence and reuse it without recomputation:

```bash
python pipeline.py export --run-dir results/test200 \
  --data-path dataset/FB15k-237 --output artifacts/test200
python pipeline.py replay --bundle artifacts/test200 \
  --output results/test200_replay
```

## 4. Paper and attribution

This repository accompanies **OASE-KGC: Ontology-Aware Subgraph Evidence Optimization for Knowledge Graph Completion**. Paper links and citation details will be added when available.

FB15k-237 and WN18RR retain their original dataset attribution and usage terms.
