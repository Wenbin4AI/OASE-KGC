"""OASE-KGC stages: train, ontology, evidence, evaluate, export, replay."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from extract_ontology import digest, load_graph, write_json
from evaluate_sample import (RankingClient, compact_record, config, file_hash,
                             full_order, ranks, save_reports, metric_group)

FORMAT = "oase-evidence-v1"
STAGES = {"train": "model/KGE_model.py", "ontology": "extract_ontology.py",
          "evidence": "evidence_pipeline.py", "evaluate": "evaluate_sample.py"}
DEFAULT_BUNDLE = "artifacts/fb15k237-sample200"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def export_bundle(run_dir, data_path, output, triple_order="htr"):
    """Publish a completed experiment without moving or modifying source results."""
    run_dir, output, data_path = Path(run_dir), Path(output), Path(data_path)
    source = read_json(run_dir / "manifest.json")
    paths = sorted((run_dir / "per_query").glob("*.json"))
    sides = ["tail", "head"] if source["prediction"] == "both" else [source["prediction"]]
    expected = {(i, s) for i in range(len(source["targets"])) for s in sides}
    if not paths or len(paths) != len(expected):
        raise ValueError("Export requires a complete experiment with all per_query checkpoints")
    for name, checksum in source["source_hashes"].items():
        data = (data_path / name).read_bytes()
        # Git may change text line endings when a Linux run is copied to Windows.
        if checksum not in {hashlib.sha256(data).hexdigest(), hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()}:
            raise ValueError(f"Dataset changed since experiment: {name}")
    entities, _, triples, _ = load_graph(data_path, source["protocol"]["filter_splits"], triple_order)
    known = set(triples) | {tuple(t) for t in source["targets"]}
    tails, heads = defaultdict(set), defaultdict(set)
    for h, r, t in known:
        tails[h, r].add(t)
        heads[t, r].add(h)
    if output.exists():
        raise ValueError("Bundle output already exists; choose a new directory")
    output.mkdir(parents=True)
    labels, seen, summaries = [], set(), []
    with gzip.open(output / "queries.jsonl.gz", "wt", encoding="utf-8") as qs, gzip.open(output / "reference_predictions.jsonl.gz", "wt", encoding="utf-8") as rs:
        for path in paths:
            record = read_json(path)
            key = (record["query_index"], record["prediction"])
            if key not in expected or key in seen or record["fingerprint"] != source["fingerprint"]:
                raise ValueError("Inconsistent source checkpoints")
            seen.add(key)
            if record["gold_triple"]["ids"] != list(source["targets"][key[0]]):
                raise ValueError("Checkpoint target differs from manifest")
            h, r, t = record["gold_triple"]["ids"]
            cids = {c["entity_id"] for c in record["candidates"]}
            order = record["full_entity_ranking"]
            if len(order) != len(set(order)) or set(order) != set(entities):
                raise ValueError("Source full ranking does not cover dataset entities")
            if len(cids) != len(record["candidates"]) or not cids <= set(order):
                raise ValueError("Invalid source candidates")
            answer_set = tails[h, r] if key[1] == "tail" else heads[t, r]
            actual_ranks = ranks(order, t if key[1] == "tail" else h, answer_set)
            if actual_ranks != (record["raw_rank"], record["filtered_rank"]):
                raise ValueError("Source metrics differ from reconstructed filtered protocol")
            uid = f"{key[0]:06d}_{key[1]}"
            evidence = {k: v for k, v in record["evidence"].items() if k != "evidence_pool"}
            query = {"id": uid, "query_index": key[0], "prediction": key[1],
                     "candidates": record["candidates"], "llm_messages": record["llm_messages"],
                     "evidence": evidence,
                     "remaining_entity_order": [e for e in order if e not in cids]}
            labels.append({"id": uid, "gold_triple": record["gold_triple"],
                           "known_answers": sorted(answer_set)})
            reference = {k: record[k] for k in ("query_index", "prediction", "gold_triple", "predicted_triple",
                         "correct", "raw_rank", "filtered_rank", "gold_in_candidates", "llm", "llm_ranked_entity_ids")}
            reference["id"] = uid
            summaries.append(compact_record(record, path))
            qs.write(json.dumps(query, ensure_ascii=False) + "\n")
            rs.write(json.dumps(reference, ensure_ascii=False) + "\n")
    write_json(output / "labels.json", labels)
    write_json(output / "reference_metrics.json", {"protocol": source["protocol"],
               "overall": metric_group(summaries, [1, 3, 10])})
    files = ["queries.jsonl.gz", "labels.json", "reference_predictions.jsonl.gz", "reference_metrics.json"]
    manifest = {"format": FORMAT, "query_count": len(seen), "dataset": source.get("dataset_name", data_path.name),
                "source_fingerprint": source["fingerprint"], "protocol": source["protocol"],
                "source_hashes": source["source_hashes"], "entity_count": len(entities),
                "reference_model": source.get("llm_config", {}).get("model"),
                "source_settings": {k: source.get(k) for k in ("top_k", "endpoint_limit", "token_budget", "encoding", "ontology_hash", "embedding_hashes")},
                "files": {name: file_hash(output / name) for name in files}}
    write_json(output / "manifest.json", manifest)
    print(f"Exported {len(seen)} queries to {output}; source results preserved.")
    return manifest


def validate_bundle(path):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if manifest.get("format") != FORMAT:
        raise ValueError("Unsupported bundle format")
    for name in ("queries.jsonl.gz", "labels.json", "reference_predictions.jsonl.gz", "reference_metrics.json"):
        if file_hash(path / name) != manifest["files"][name]:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    labels = read_json(path / "labels.json")
    label_map = {r["id"]: r for r in labels}
    seen = set()
    for q in lines(path / "queries.jsonl.gz"):
        if q["id"] in seen or q["id"] not in label_map:
            raise ValueError("Duplicate or unmatched query ID")
        seen.add(q["id"])
        cids = [c["entity_id"] for c in q["candidates"]]
        order = cids + q["remaining_entity_order"]
        if not cids or len(order) != len(set(order)) or len(order) != manifest["entity_count"]:
            raise ValueError("Invalid candidate/full-ranking coverage")
        if [c["index"] for c in q["candidates"]] != list(range(len(cids))):
            raise ValueError("Candidate indices must be contiguous")
        payload = json.loads(q["llm_messages"][1]["content"])
        if set(payload) != {"query", "answer_type_requirements", "evidence", "candidates"} or payload["candidates"] != q["candidates"]:
            raise ValueError("Unexpected prompt fields or candidate mismatch")
    if len(seen) != manifest["query_count"] or len(labels) != len(label_map) or seen != set(label_map):
        raise ValueError("Bundle count mismatch")
    return manifest


def replay(args):
    bundle, out = Path(args.bundle), Path(args.output)
    manifest = validate_bundle(bundle)
    if args.limit < 0:
        raise ValueError("limit must be nonnegative")
    if args.validate_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    cfg = config(args.config)
    labels = {r["id"]: r for r in read_json(bundle / "labels.json")}
    total = min(args.limit or manifest["query_count"], manifest["query_count"])
    run = {"bundle_manifest_hash": file_hash(bundle / "manifest.json"), "version": FORMAT,
           "limit": args.limit, "evidence_mode": args.evidence_mode,
           "llm_config": {k: v for k, v in cfg.items() if k != "api_key"},
           "protocol": {**manifest["protocol"], "fixed_evidence_replay": True, "evidence_mode": args.evidence_mode}}
    fingerprint = digest(run)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() and read_json(out / "manifest.json")["fingerprint"] != fingerprint:
        raise ValueError("Replay settings changed; choose another output directory")
    write_json(out / "manifest.json", {"fingerprint": fingerprint, **run})
    checkpoints = out / "per_query"
    checkpoints.mkdir(exist_ok=True)
    client = RankingClient(cfg)
    records, failed = [], False
    try:
        for index, q in enumerate(lines(bundle / "queries.jsonl.gz")):
            if index >= total:
                break
            path = checkpoints / f"{index:06d}.json"
            if path.exists():
                record = read_json(path)
                if record["fingerprint"] != fingerprint:
                    raise ValueError("Replay checkpoint mismatch")
                records.append(compact_record(record, path))
                continue
            if not cfg["api_key"]:
                raise ValueError("Set the API key environment variable from api_key_env")
            messages = copy.deepcopy(q["llm_messages"])
            if args.evidence_mode == "none":
                payload = json.loads(messages[1]["content"])
                payload["evidence"] = ""
                messages[1]["content"] = json.dumps(payload, ensure_ascii=False)
            llm = client.rank(messages, len(q["candidates"]))
            if "error" in llm:
                write_json(out / "last_error.json", {"id": q["id"], "llm": llm})
                failed = True
                break
            label = labels[q["id"]]  # gold and filtering are never sent to LLM
            h, r, t = label["gold_triple"]["ids"]
            gold = t if q["prediction"] == "tail" else h
            order = full_order(q["remaining_entity_order"], q["candidates"], llm["ranking"])
            raw, filtered = ranks(order, gold, set(label["known_answers"]))
            predicted = order[0]
            pred_name = next(c["label"] for c in q["candidates"] if c["entity_id"] == predicted)
            predicted_triple = copy.deepcopy(label["gold_triple"])
            position = 2 if q["prediction"] == "tail" else 0
            predicted_triple["ids"][position], predicted_triple["labels"][position] = predicted, pred_name
            record = {"fingerprint": fingerprint, "query_index": q["query_index"], "prediction": q["prediction"],
                      "bundle_query_id": q["id"], "gold_triple": label["gold_triple"], "predicted_triple": predicted_triple,
                      "correct": predicted == gold, "raw_rank": raw, "filtered_rank": filtered,
                      "gold_in_candidates": any(c["entity_id"] == gold for c in q["candidates"]),
                      "candidates": q["candidates"], "llm_messages": messages, "llm": llm,
                      "llm_ranked_entity_ids": [q["candidates"][i]["entity_id"] for i in llm["ranking"]],
                      "full_entity_ranking": order, "evidence": q["evidence"], "evidence_mode": args.evidence_mode}
            write_json(path, record)
            records.append(compact_record(record, path))
            print(f"{len(records)}/{total}: raw_rank={raw}, filtered_rank={filtered}, exact_top1_correct={predicted == gold}", flush=True)
    finally:
        print(json.dumps(save_reports(out, records, total, [1, 3, 10], run), ensure_ascii=False, indent=2))
    return 1 if failed or len(records) != total else 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in STAGES:
        return subprocess.call([sys.executable, str(Path(__file__).parent / STAGES[argv[0]]), *argv[1:]])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="Copy a completed evaluation to a portable bundle")
    export.add_argument("--run-dir", required=True)
    export.add_argument("--data-path", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--triple-order", choices=["htr", "hrt"], default="htr")
    rp = sub.add_parser("replay", help="LLM-only evaluation: no KG, ontology or weights needed")
    rp.add_argument("--bundle", default=DEFAULT_BUNDLE)
    rp.add_argument("--config", default="ontology_config.local.json")
    rp.add_argument("--output", default="results/replay")
    rp.add_argument("--limit", type=int, default=0)
    rp.add_argument("--evidence-mode", choices=["full", "none"], default="full")
    rp.add_argument("--validate-only", action="store_true")
    for stage in STAGES:
        sub.add_parser(stage, help=f"Forward arguments to {STAGES[stage]}")
    args = parser.parse_args(argv)
    if args.command == "export":
        export_bundle(args.run_dir, args.data_path, args.output, args.triple_order)
        return 0
    return replay(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
