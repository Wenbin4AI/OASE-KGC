"""General KG evaluation: full splits, random subsets, or explicit HRT targets.
Uses training-only evidence and ontology, LLM reranking, and full-entity metrics.
Targets overlapping training remain diagnostic rather than held-out evaluation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
import random
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from urllib import error, parse, request

import numpy as np

from extract_ontology import digest, load_graph, write_json
from evidence_pipeline import Embeddings, EvidencePipeline, Ontology
from src.kg_loader import KGLoader

VERSION = "general-evaluation-2"


def select_targets(args, source, train, entities, relations):
    """Preserve split row order and source indices; random subsets are reproducible."""
    if args.triples and args.split:
        raise ValueError("Use either --triples or --split")
    mode = "custom" if args.triples else (args.split or "test")
    rows = []
    if mode in {"custom", "ontology-sample"}:
        path = Path(args.triples) if args.triples else Path(args.ontology).with_name("selected_triples.json")
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, list) or not values:
            raise ValueError("Targets must be a nonempty JSON array")
        if any(not isinstance(t, list) or len(t) != 3 or any(type(v) is not int for v in t) for t in values):
            raise ValueError("Targets must contain integer [head, relation, tail] arrays")
        if len({tuple(t) for t in values}) != len(values):
            raise ValueError("Custom/sample target triples must be distinct")
        if mode == "ontology-sample" and digest(sorted(values)) != source["manifest"].get("scope_hash"):
            raise ValueError("selected_triples.json does not match ontology scope")
        rows = [{"triple": tuple(t), "source": mode, "source_index": i} for i, t in enumerate(values)]
    else:
        seen = set()
        for split in (["train", "valid", "test"] if mode == "all" else [mode]):
            order = source.get("manifest", {}).get("triple_order", "htr")
            load_graph(args.data_path, [split], order)  # validates IDs, rows and count
            values = KGLoader(args.data_path)._read_id_triples(Path(args.data_path) / (split + "2id.txt"))
            for i, (a, b, c) in enumerate(values):
                tri = (a, c, b) if order == "htr" else (a, b, c)
                if mode == "all" and tri in seen:
                    continue
                seen.add(tri)
                rows.append({"triple": tri, "source": split, "source_index": i})
    for row in rows:
        h, r, t = row["triple"]
        if h not in entities or t not in entities or r not in relations:
            raise ValueError("Target contains an unknown entity/relation ID")
    available = len(rows)
    # Keep old sample diagnostic's entire target-set exclusion for smoke tests.
    sample_excluded = {row["triple"] for row in rows} if mode == "ontology-sample" else set()
    if args.sample_size:
        if args.sample_size > available:
            raise ValueError(f"sample-size {args.sample_size} exceeds {available} available triples")
        selected_indices = sorted(random.Random(args.seed).sample(range(available), args.sample_size))
        rows = [rows[i] for i in selected_indices]
    elif args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("No target triples selected")
    excluded = (sample_excluded or {row["triple"] for row in rows}) & set(train)
    return rows, excluded, mode, available


def compact_record(record, path):
    """Keep only metrics and a disk reference in memory for large evaluations."""
    return {"_path": str(path), **{k: record[k] for k in
            ("query_index", "prediction", "correct", "raw_rank", "filtered_rank", "gold_in_candidates")},
            "llm": {"fallback_used": record["llm"]["fallback_used"]}}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_ranking(text, count):
    """Return original parsed list, valid permutation prefix, completeness flag."""
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        value = "\n".join(value.splitlines()[1:-1])
    try:
        obj = json.loads(value)
        original = obj.get("ranking") if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        original = None
    prefix = []
    if isinstance(original, list):
        for idx in original:
            if type(idx) is int and 0 <= idx < count and idx not in prefix:
                prefix.append(idx)
    complete = isinstance(original, list) and len(original) == count and len(prefix) == count
    return original, prefix, complete


class RankingClient:
    def __init__(self, cfg):
        self.cfg = cfg

    def rank(self, messages, count):
        cfg = self.cfg
        attempts, best_prefix, last_parsed = [], [], None
        endpoint = cfg["base_url"].rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        for attempt in range(cfg["retries"] + 1):
            body = {"model": cfg["model"], "messages": messages, "temperature": 0,
                    "max_tokens": cfg["max_tokens"]}
            if cfg["json_mode"]:
                body["response_format"] = {"type": "json_object"}
            try:
                req = request.Request(endpoint, data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json", "Authorization": "Bearer " + cfg["api_key"]})
                with request.urlopen(req, timeout=cfg["timeout"]) as response:
                    response_body = json.load(response)
                choice = response_body["choices"][0]
                content = choice["message"].get("content") or ""
                if not isinstance(content, str):
                    raise ValueError("Expected string message content")
                last_parsed, prefix, valid = parse_ranking(content, count)
                attempts.append({"response": content, "parsed_ranking": last_parsed,
                                 "finish_reason": choice.get("finish_reason"),
                                 "usage": response_body.get("usage"), "valid": valid})
                best_prefix = prefix
                if valid and choice.get("finish_reason") != "length":
                    return {"ranking": prefix, "fallback_used": False, "attempts": attempts}
            except error.HTTPError as exc:
                attempts.append({"error": f"HTTP {exc.code}"})
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    return {"error": f"HTTP {exc.code}", "attempts": attempts}
            except (error.URLError, TimeoutError, OSError):
                attempts.append({"error": "Network error or timeout"})
            except (ValueError, KeyError, IndexError, TypeError):
                attempts.append({"error": "Invalid API response envelope"})
            if attempt < cfg["retries"]:
                time.sleep(min(2 ** attempt, 30))
        if not any("response" in a for a in attempts):
            return {"error": "No LLM response after retries", "attempts": attempts}
        # Paper A.3 fallback: retain valid indices, append missing in retrieval order.
        completed = best_prefix + [i for i in range(count) if i not in best_prefix]
        return {"ranking": completed, "fallback_used": True, "attempts": attempts}


def full_order(base_order, candidates, ranking):
    promoted = [candidates[i]["entity_id"] for i in ranking]
    seen = set(promoted)
    return promoted + [eid for eid in base_order if eid not in seen]


def ranks(order, gold, known):
    raw = order.index(gold) + 1
    filtered = 1 + sum(e not in known for e in order[:raw - 1])
    return raw, filtered


def metric_group(records, ks):
    count = len(records)
    result = {"count": count}
    for name, field in (("raw", "raw_rank"), ("filtered", "filtered_rank")):
        values = [r[field] for r in records]
        result[name] = {"MR": sum(values) / count if count else None,
                        "MRR": sum(1 / v for v in values) / count if count else None,
                        **{f"Hits@{k}": sum(v <= k for v in values) / count if count else None for k in ks}}
    result["exact_top1_accuracy"] = sum(r["correct"] for r in records) / count if count else None
    result["candidate_recall"] = sum(r["gold_in_candidates"] for r in records) / count if count else None
    result["fallback_count"] = sum(r["llm"]["fallback_used"] for r in records)
    return result


def save_reports(out, records, total, ks, manifest):
    # Sort samples only. Never reorder a model ranking using knowledge of gold.
    ordered = sorted(records, key=lambda r: (not r["correct"], r["query_index"], r["prediction"]))
    metrics = {"status": "complete" if len(records) == total else "partial",
               "expected_queries": total, "completed_queries": len(records),
               "protocol": manifest["protocol"], "overall": metric_group(records, ks),
               "by_prediction": {s: metric_group([r for r in records if r["prediction"] == s], ks)
                                 for s in ("tail", "head") if any(r["prediction"] == s for r in records)}}
    write_json(out / "metrics.json", metrics)
    temp = out / "predictions_sorted.csv.tmp"
    json_temp = out / "predictions_sorted.json.tmp"
    with temp.open("w", encoding="utf-8-sig", newline="") as stream, json_temp.open("w", encoding="utf-8") as json_stream:
        json_stream.write("[\n")
        writer = csv.writer(stream)
        writer.writerow(["query_index", "prediction", "correct", "gold_triple", "predicted_triple",
                         "raw_rank", "filtered_rank", "gold_in_candidates", "fallback_used", "evidence"])
        for i, summary in enumerate(ordered):
            row = json.loads(Path(summary["_path"]).read_text(encoding="utf-8")) if "_path" in summary else summary
            if i:
                json_stream.write(",\n")
            json.dump(row, json_stream, ensure_ascii=False)
            writer.writerow([row["query_index"], row["prediction"], row["correct"],
                             json.dumps(row["gold_triple"]["labels"], ensure_ascii=False),
                             json.dumps(row["predicted_triple"]["labels"], ensure_ascii=False),
                             row["raw_rank"], row["filtered_rank"], row["gold_in_candidates"],
                             row["llm"]["fallback_used"], row["evidence"]["evidence_text"]])
        json_stream.write("\n]\n")
    temp.replace(out / "predictions_sorted.csv")
    json_temp.replace(out / "predictions_sorted.json")
    return metrics


def config(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    cfg = {k: raw.get(k, default) for k, default in {
        "base_url": "", "model": "", "api_key_env": "LLM_API_KEY", "timeout": 120,
        "retries": 3, "max_tokens": 4096, "json_mode": False}.items()}
    cfg["base_url"] = os.getenv("LLM_BASE_URL") or cfg["base_url"]
    cfg["model"] = os.getenv("LLM_MODEL") or cfg["model"]
    url = parse.urlparse(cfg["base_url"])
    if url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password or url.query or url.fragment:
        raise ValueError("Invalid base_url")
    if not cfg["model"]:
        raise ValueError("model is required")
    for key in ("timeout", "max_tokens", "retries"):
        if type(cfg[key]) is not int or cfg[key] < (0 if key == "retries" else 1):
            raise ValueError(f"Invalid {key}")
    if not isinstance(cfg["api_key_env"], str) or cfg["api_key_env"].startswith("sk-"):
        raise ValueError("api_key_env must be an environment variable name, e.g. LLM_API_KEY")
    cfg["api_key"] = os.getenv(cfg["api_key_env"], "")
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default="dataset/FB15k-237")
    p.add_argument("--ontology", help="Default: import/ontology/DATASET/all/ontology.json")
    p.add_argument("--split", choices=["train", "valid", "test", "all", "ontology-sample"],
                   help="Default: test; all = deduplicated train+valid+test")
    p.add_argument("--triples", help="Custom HRT JSON array; mutually exclusive with split")
    p.add_argument("--embeddings", required=True)
    p.add_argument("--config", default="ontology_config.local.json")
    p.add_argument("--prediction", choices=["tail", "head", "both"], default="tail")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--endpoint-limit", type=int, default=None)
    p.add_argument("--token-budget", type=int, default=3000)
    p.add_argument("--encoding", help="e.g. o200k_base, requires tiktoken")
    p.add_argument("--tokenizer", help="Local Hugging Face tokenizer")
    p.add_argument("--hits-k", default="1,3,10")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=0, help="First N target triples, 0=all")
    selection.add_argument("--sample-size", type=int, default=0, help="Random N triples, 0=all")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", default="results/kg_prediction")
    p.add_argument("--dry-run", action="store_true", help="Validate inputs; no API calls")
    args = p.parse_args(argv)
    ks = sorted(set(map(int, args.hits_k.split(","))))
    if min(ks) < 1 or args.top_k < 1 or args.token_budget < 1 or args.limit < 0 or args.sample_size < 0:
        raise ValueError("Invalid numerical argument")
    if args.endpoint_limit is not None and args.endpoint_limit < 0:
        raise ValueError("endpoint-limit must be nonnegative")
    if args.tokenizer and args.encoding:
        raise ValueError("Choose tokenizer or encoding, not both")
    cfg = config(args.config)
    if args.ontology is None:
        args.ontology = str(Path("import/ontology") / Path(args.data_path).name /
                            ("sample" if args.split == "ontology-sample" else "all") / "ontology.json")
    source = json.loads(Path(args.ontology).read_text(encoding="utf-8"))
    order = source.get("manifest", {}).get("triple_order", "htr")
    entities, relations, train, hashes = load_graph(args.data_path, ["train"], order)
    data = SimpleNamespace(dataset_name=Path(args.data_path).name, entities=entities, relations=relations)
    ontology = Ontology(data, source)
    ontology.validate_provenance(args.data_path)
    target_rows, excluded, target_mode, available = select_targets(args, source, train, entities, relations)
    targets = [row["triple"] for row in target_rows]
    data.train_triples = [SimpleNamespace(h_id=h, r_id=r, t_id=t) for h, r, t in train if (h, r, t) not in excluded]
    known = set(train)
    known.update(targets)  # custom target facts also contribute to filtered answers
    filter_splits = ["train"]
    for split in ("valid", "test"):
        if (Path(args.data_path) / (split + "2id.txt")).exists():
            _, _, extra, extra_hashes = load_graph(args.data_path, [split], order)
            known.update(extra)
            hashes.update(extra_hashes)
            filter_splits.append(split)
    tails, heads = defaultdict(set), defaultdict(set)
    for h, r, t in known:
        tails[h, r].add(t)
        heads[t, r].add(h)
    train_set = set(train)
    train_overlap = sum(t in train_set for t in targets)
    emb_path = Path(args.embeddings)
    emb_files = [emb_path / name for name in ("entity_embedding.npy", "relation_embedding.npy", "config.json")] if emb_path.is_dir() else [emb_path]
    manifest = {"version": VERSION, "dataset_name": data.dataset_name, "targets": targets,
                "target_sources": target_rows, "target_mode": target_mode, "available_triples": available,
                "selection": {"sample_size": args.sample_size, "limit": args.limit, "seed": args.seed},
                "excluded_target_count": len(excluded), "excluded_target_hash": digest(sorted(excluded)),
                "source_hashes": hashes, "ontology_hash": file_hash(args.ontology),
                "embedding_hashes": {f.name: file_hash(f) for f in emb_files},
                "prediction": args.prediction, "top_k": args.top_k, "endpoint_limit": args.endpoint_limit,
                "token_budget": args.token_budget, "encoding": args.encoding, "tokenizer": args.tokenizer,
                "hits_k": ks, "llm_config": {k: v for k, v in cfg.items() if k != "api_key"},
                "protocol": {"evaluation": "training_overlap_diagnostic" if train_overlap else "held_out_targets_training_evidence",
                    "all_target_edges_removed_from_evidence": True, "training_overlap_count": train_overlap,
                    "ontology_source": "train", "embedding_training_provenance": "caller_must_verify_train_only",
                    "filter_splits": filter_splits, "custom_targets_in_filter": target_mode == "custom",
                    "ranking": "LLM candidates first; remaining entities in RotatE order",
                    "correct_definition": "unfiltered top1 equals this target entity"}}
    if args.tokenizer:
        manifest["tokenizer_hashes"] = {str(f.relative_to(args.tokenizer)): file_hash(f)
                                        for f in sorted(Path(args.tokenizer).rglob("*")) if f.is_file()}
    fingerprint = digest(manifest)
    embeddings = Embeddings(emb_path, data)
    tokenizer = None
    if args.encoding:
        import tiktoken
        enc = tiktoken.get_encoding(args.encoding)
        tokenizer = SimpleNamespace(encode=lambda text, **kw: enc.encode(text, disallowed_special=()))
    elif args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    sides = ["tail", "head"] if args.prediction == "both" else [args.prediction]
    total = len(targets) * len(sides)
    if args.dry_run:
        print(json.dumps({"target_mode": target_mode, "available_triples": available,
                          "triples": len(targets), "queries": total, "evidence_edges": len(data.train_triples),
                          "protocol": manifest["protocol"]}, ensure_ascii=False, indent=2))
        return 0
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        previous = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Inputs/settings changed; use a different --output directory")
    else:
        write_json(out / "manifest.json", {"fingerprint": fingerprint, **manifest})
    per_query = out / "per_query"
    per_query.mkdir(exist_ok=True)
    pipelines = {side: EvidencePipeline(data, embeddings, ontology, args.token_budget, tokenizer, reverse=side == "head") for side in sides}
    client = RankingClient(cfg)
    records, failed = [], False

    def triple_record(tri):
        h, r, t = tri
        return {"ids": list(tri), "labels": [entities[h].label, relations[r].label, entities[t].label]}

    try:
        for index, (h, r, t) in enumerate(targets):
            for side in sides:
                path = per_query / f"{index:04d}_{side}.json"
                if path.exists():
                    record = json.loads(path.read_text(encoding="utf-8"))
                    if record.get("fingerprint") != fingerprint:
                        raise ValueError("Checkpoint fingerprint mismatch")
                    records.append(compact_record(record, path))
                    continue
                if not cfg["api_key"]:
                    raise ValueError("Set the API key environment variable configured by api_key_env")
                pipeline = pipelines[side]
                anchor, gold = (h, t) if side == "tail" else (t, h)
                evidence = pipeline.run_query(anchor, r)
                endpoint_limit = args.endpoint_limit if args.endpoint_limit is not None else (20 if data.dataset_name == "WN18RR" else 50)
                candidates = pipeline.candidates(anchor, r, evidence["evidence_pool"], args.top_k, endpoint_limit)
                messages = pipeline.messages(evidence, candidates)
                llm = client.rank(messages, len(candidates))
                if "error" in llm:
                    write_json(out / "last_error.json", {"query_index": index, "prediction": side, "llm": llm})
                    print(f"Stopped at {index}/{side}: {llm['error']}. Rerun the same command to resume.")
                    failed = True
                    break
                ids = np.array(sorted(entities))
                distances = np.abs(pipeline.emb.entity[anchor] * pipeline.emb.relation[r] - pipeline.emb.entity[ids]).sum(axis=1)
                base_order = ids[np.lexsort((ids, distances))].tolist()
                final_order = full_order(base_order, candidates, llm["ranking"])
                raw_rank, filtered_rank = ranks(final_order, gold, tails[anchor, r] if side == "tail" else heads[anchor, r])
                predicted = final_order[0]
                prediction = (h, r, predicted) if side == "tail" else (predicted, r, t)
                record = {"fingerprint": fingerprint, "query_index": index, "prediction": side,
                    "target_source": target_rows[index]["source"], "source_index": target_rows[index]["source_index"],
                    "gold_triple": triple_record((h, r, t)), "predicted_triple": triple_record(prediction),
                    "correct": predicted == gold, "raw_rank": raw_rank, "filtered_rank": filtered_rank,
                    "reciprocal_rank": 1 / filtered_rank,
                    "gold_in_candidates": any(c["entity_id"] == gold for c in candidates),
                    "llm": llm, "llm_ranked_entity_ids": [candidates[i]["entity_id"] for i in llm["ranking"]],
                    "full_entity_ranking": final_order, "candidates": candidates,
                    "llm_messages": messages, "evidence": evidence}
                if side == "head":
                    evidence["query_tail_id"] = evidence.pop("query_head_id")
                    evidence["query_tail_label"] = evidence.pop("query_head_label")
                evidence["provenance"].update(excluded_target_count=len(excluded), training_sample_diagnostic=bool(train_overlap),
                    cost_mode=args.encoding or ("local_tokenizer" if tokenizer else "utf8_bytes_upper_bound"))
                write_json(path, record)
                records.append(compact_record(record, path))
                print(f"{len(records)}/{total} {side}: rank={filtered_rank}, correct={predicted == gold}", flush=True)
            if failed:
                break
    finally:
        metrics = save_reports(out, records, total, ks, manifest)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
