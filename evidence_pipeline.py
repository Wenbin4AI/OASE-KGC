"""OASE-KGC sections 4.1--4.3, equations (2)--(7), (11), (14).

Examples (run from the project root)::

    python evidence_pipeline.py --data-path dataset/FB15k-237 --embeddings import/KGE_model/FB15k-237 --limit 10
    python evidence_pipeline.py --data-path dataset/WN18RR --embeddings import/KGE_model/WN18RR --split test

Embeddings directory: entity_embedding.npy, relation_embedding.npy, config.json
(exported by model/KGE_model.py). Alternatively pass its checkpoint.pt; this
requires torch and reads tensors using weights_only=True. No automatic training,
random embeddings, LLM calls, or access to gold answers during extraction.

Optional --ontology JSON format::

    {"subclass_of": [["DoctoralResearcher", "Researcher"]],
     "disjoint_with": [["City", "Researcher"]],
     "satisfiable_classes": ["TeachingResearcher"],
     "entity_classes": {"0": ["Person", "Researcher"]},
     "relation_schema": {"0": {"domain": "Person", "range": "Country"}}}

Entity classes augment entity.json; schema overrides relation.json. Class names
are case-folded with normalized whitespace, without fuzzy synonym matching.
Schema lists mean conjunction (paper Eq. 4); {"any_of": [...]} means union.
Overlap requires an explicitly satisfiable common subclass. Missing axioms stay
unknown. *_candidates frequency lists are NOT logical schema constraints.
Ontology provenance must be checked by the caller: existing JSON does not record
whether its construction used training data only.

Output JSONL contains evidence_text ready for a later prompt, selected_evidence,
full calibrated pool, score components, schema checks, and graph provenance.
Budget covers serialized evidence only; prompt/candidates are budgeted separately.
The default UTF-8-byte cost is conservative for byte-level BPE tokenizers; use
--tokenizer PATH with a local HF tokenizer for the intended LLM. A model-specific
context-window check remains the responsibility of the downstream predictor.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

from src.kg_loader import KGLoader


def names(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [s for item in value for s in names(item)]
    raise ValueError(f"Expected a class name or list, got {value!r}")


def canonical(value):
    return " ".join(value.casefold().split())


class Ontology:
    def __init__(self, dataset, supplement=None):
        extra = supplement or {}
        self.classes = {}
        for eid, entity in dataset.entities.items():
            raw = entity.raw or {}
            values = names(entity.classname) + names(raw.get("classes"))
            values += names(extra.get("entity_classes", {}).get(str(eid)))
            self.classes[eid] = frozenset(map(canonical, values))
        self.schema = {
            rid: extra.get("relation_schema", {}).get(str(rid),
                     {"domain": r.domain, "range": r.range})
            for rid, r in dataset.relations.items()
        }
        self.parents = defaultdict(set)
        self.known = set().union(*self.classes.values()) if self.classes else set()
        for child, parent in extra.get("subclass_of", []):
            child, parent = canonical(child), canonical(parent)
            self.parents[child].add(parent)
            self.known.update((child, parent))
        self.disjoint = {frozenset(map(canonical, pair))
                         for pair in extra.get("disjoint_with", [])}
        self.satisfiable = set(map(canonical, extra.get("satisfiable_classes", [])))

    @lru_cache(maxsize=None)
    def ancestors(self, cls):
        found, todo = {cls}, [cls]
        while todo:
            for parent in self.parents[todo.pop()]:
                if parent not in found:
                    found.add(parent)
                    todo.append(parent)
        return frozenset(found)

    def is_disjoint(self, a, b):
        return any(frozenset((x, y)) in self.disjoint
                   for x in self.ancestors(a) for y in self.ancestors(b))

    @lru_cache(maxsize=None)
    def atomic(self, eid, target):
        classes = self.classes.get(eid, ())
        if any(target in self.ancestors(c) for c in classes):
            return 1.0, "match_or_subclass"
        if any(c != target and c in self.ancestors(target) for c in classes):
            return 0.9, "superclass"
        if any(target in self.ancestors(s) and c in self.ancestors(s)
               and c not in self.ancestors(target) and target not in self.ancestors(c)
               and not any(self.is_disjoint(a, b) for a in self.ancestors(s)
                           for b in self.ancestors(s))
               for c in classes for s in self.satisfiable):
            return 0.65, "supported_overlap"
        if any(self.is_disjoint(c, target) for c in classes):
            return 0.1, "disjoint"
        return 0.6, "unknown"

    def compatibility(self, eid, expression):
        if isinstance(expression, dict):
            if set(expression) != {"any_of"} or not expression["any_of"]:
                raise ValueError("Schema expressions support only nonempty any_of")
            return max((self.compatibility(eid, x) for x in expression["any_of"]),
                       key=lambda x: x[0])
        if isinstance(expression, list):
            results = [self.compatibility(eid, x) for x in expression]
            return (math.prod(x[0] for x in results), "conjunction") if results else (0.6, "unknown")
        if expression is None or expression == "":
            return 0.6, "unknown"
        return self.atomic(eid, canonical(expression))

    @lru_cache(maxsize=None)
    def triple(self, h, r, t):
        schema = self.schema[r]
        left = self.compatibility(h, schema.get("domain"))
        right = self.compatibility(t, schema.get("range"))
        return left[0] * right[0], {"domain": left, "range": right}


class Embeddings:
    def __init__(self, path, dataset):
        path = Path(path)
        if path.is_dir():
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            entity = np.load(path / "entity_embedding.npy", allow_pickle=False)
            relation = np.load(path / "relation_embedding.npy", allow_pickle=False)
            scale = (float(config["gamma"]) + 2.0) / int(config["hidden_dim"])
        else:
            import torch
            state = torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]
            entity = state["entity_embedding"].numpy()
            relation = state["relation_embedding"].numpy()
            scale = float(state["embedding_range"].item())
        if entity.ndim != 2 or relation.ndim != 2 or entity.shape[1] != 2 * relation.shape[1]:
            raise ValueError("Expected RotatE entity [N,2d] and phase relation [R,d]")
        if scale <= 0 or not np.isfinite(entity).all() or not np.isfinite(relation).all():
            raise ValueError("Invalid or non-finite RotatE embeddings")
        if max(dataset.entities) >= len(entity) or max(dataset.relations) >= len(relation):
            raise ValueError("Embedding IDs do not cover dataset IDs")
        d = relation.shape[1]
        self.entity = entity[:, :d].astype(np.float64) + 1j * entity[:, d:]
        self.relation = np.exp(1j * relation.astype(np.float64) * np.pi / scale)


def defaults(dataset_name):
    if dataset_name == "WN18RR":
        return dict(hops=2, beam=20, neighbors=80, paths=50, alpha_n=0.5,
                    alpha_p=0.6, tau_t=0.5, tau_e=0.8, decay=0.3,
                    theta=0.6, weights=(0.35, 0.35, 0.30))
    return dict(hops=3, beam=30, neighbors=150, paths=100, alpha_n=0.4,
                alpha_p=0.5, tau_t=0.8, tau_e=1.2, decay=0.5,
                theta=0.5, weights=(0.40, 0.35, 0.25))


class EvidencePipeline:
    def __init__(self, dataset, embeddings, ontology, budget=3000, tokenizer=None):
        self.data, self.emb, self.ontology = dataset, embeddings, ontology
        self.config = defaults(dataset.dataset_name)
        self.budget = budget
        self.tokenizer = tokenizer
        self.adj = defaultdict(list)
        # The only graph used for extraction. Do not pass valid/test to this index.
        self.train = {(x.h_id, x.r_id, x.t_id) for x in dataset.train_triples}
        for h, r, t in sorted(self.train):
            self.adj[h].append((h, r, t))

    @staticmethod
    def cosine(a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.clip(np.real(np.vdot(a, b)) / denom, -1, 1)) if denom else 0.0

    def structural(self, path, head, relation):
        p = self.config
        query = self.emb.relation[relation]
        composed = np.prod([self.emb.relation[r] for _, r, _ in path], axis=0)
        relevance = (1 + self.cosine(composed, query)) / 2
        if len(path) == 1:
            h, r, t = path[0]
            distance = np.linalg.norm(self.emb.entity[h] * self.emb.relation[r] - self.emb.entity[t])
            reliability = math.exp(-float(distance) / p["tau_t"])
            score = p["alpha_n"] * relevance + (1 - p["alpha_n"]) * reliability
            decay = 1.0
        else:
            distance = np.linalg.norm(self.emb.entity[head] * query - self.emb.entity[path[-1][2]])
            reliability = math.exp(-float(distance) / p["tau_e"])
            decay = math.exp(-p["decay"] * (len(path) - 1))
            score = decay * (p["alpha_p"] * relevance + (1 - p["alpha_p"]) * reliability)
        return float(score), dict(relation_relevance=relevance, reliability=reliability, length_decay=decay)

    def extract(self, head, relation):
        p = self.config
        # Mask all direct query-relation edges, independently of any gold tail.
        def outgoing(node):
            return [e for e in self.adj[node] if not (e[0] == head and e[1] == relation)]
        one = [(e,) for e in outgoing(head)]
        rank = lambda path: (-self.structural(path, head, relation)[0], path)
        pool = sorted(one, key=rank)[:p["neighbors"]]
        prefixes = sorted(one, key=rank)[:p["beam"]]
        paths = []
        for _ in range(2, p["hops"] + 1):
            extended = []
            for prefix in prefixes:
                visited = {head} | {e[2] for e in prefix}
                for edge in outgoing(prefix[-1][2]):
                    if edge[2] not in visited:
                        extended.append(prefix + (edge,))
            prefixes = sorted(set(extended), key=rank)[:p["beam"]]
            paths.extend(prefixes)
        pool.extend(sorted(set(paths), key=rank)[:p["paths"]])
        return pool

    def text(self, path):
        # JSON quoting prevents labels containing newlines from changing serialization.
        label = lambda x: json.dumps(x, ensure_ascii=False)
        parts = [label(self.data.entities[path[0][0]].label)]
        for _, r, t in path:
            parts.append(f" --[{label(self.data.relations[r].label)}]--> {label(self.data.entities[t].label)}")
        return "- " + "".join(parts) + "\n"

    def cost(self, text):
        if self.tokenizer is None:
            return len(text.encode("utf-8"))
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def features(self, item):
        edges = item["triples"]
        return ({r for _, r, _ in edges}, {e for h, _, t in edges for e in (h, t)}, set(edges))

    def select(self, pool):
        if not pool:
            return [], 0.0
        features = [self.features(u) for u in pool]
        universes = [set().union(*(f[g] for f in features)) for g in range(3)]
        weights, theta = self.config["weights"], self.config["theta"]
        def gain(i, coverage):
            v = pool[i]["quality"]
            return theta * v / len(pool) + (1 - theta) * sum(
                weights[g] / len(universes[g]) * sum(max(0, v - coverage[g].get(f, 0)) for f in features[i][g])
                for g in range(3))
        coverage = [{}, {}, {}]
        remaining = set(range(len(pool)))
        chosen, spent, objective = [], 0, 0.0
        while remaining:
            feasible = [i for i in sorted(remaining) if spent + pool[i]["cost"] <= self.budget]
            if not feasible:
                break
            i = max(feasible, key=lambda j: (gain(j, coverage) / pool[j]["cost"], -j))
            delta = gain(i, coverage)
            if delta <= 0:
                break
            chosen.append(i)
            remaining.remove(i)
            spent += pool[i]["cost"]
            objective += delta
            for g in range(3):
                for feature in features[i][g]:
                    coverage[g][feature] = max(coverage[g].get(feature, 0), pool[i]["quality"])
        empty = [{}, {}, {}]
        feasible = [i for i in range(len(pool)) if pool[i]["cost"] <= self.budget]
        if feasible:
            best = max(feasible, key=lambda i: (gain(i, empty), -i))
            if gain(best, empty) > objective:
                chosen, objective = [best], gain(best, empty)
        return [pool[i] for i in chosen], objective

    def run_query(self, head, relation):
        if head not in self.data.entities or relation not in self.data.relations:
            raise ValueError(f"Unknown query IDs: {(head, relation)}")
        pool = []
        for path in self.extract(head, relation):
            score, components = self.structural(path, head, relation)
            checks = [self.ontology.triple(*edge) for edge in path]
            eta = math.prod(check[0] for check in checks)
            text = self.text(path)
            pool.append(dict(kind="one_hop" if len(path) == 1 else "path",
                             triples=path, text=text, cost=max(1, self.cost(text)),
                             structural_score=score, score_components=components,
                             ontology_weight=eta, schema_checks=[x[1] for x in checks],
                             quality=score * eta, terminal_entity_id=path[-1][2]))
        selected, objective = self.select(pool)
        text = "".join(u["text"] for u in selected)
        # BPE boundaries may change when units are joined. Verify complete serialization.
        while selected and self.cost(text) > self.budget:
            selected = selected[:-1]
            text = "".join(u["text"] for u in selected)
            objective = None  # singleton/greedy value no longer describes this fallback
        return dict(query_head_id=head, query_relation_id=relation,
                    query_head_label=self.data.entities[head].label,
                    query_relation_label=self.data.relations[relation].label,
                    evidence_text=text, selected_evidence=selected, evidence_pool=pool,
                    evidence_cost=self.cost(text), token_budget=self.budget,
                    objective=objective, config=self.config,
                    provenance={"graph_split": "train", "gold_used": False,
                                "direct_query_edges_masked": True,
                                "cost_mode": "local_tokenizer" if self.tokenizer else "utf8_bytes_upper_bound",
                                "training_triples": len(self.train)})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", default="dataset/FB15k-237")
    parser.add_argument("--embeddings", required=True, help="RotatE export directory or checkpoint.pt")
    parser.add_argument("--ontology", help="Optional supplementary ontology JSON")
    parser.add_argument("--tokenizer", help="Local Hugging Face tokenizer directory")
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--queries", help="JSONL with query_head_id and query_relation_id (tail ignored)")
    parser.add_argument("--limit", type=int, default=0, help="0 processes all queries")
    parser.add_argument("--token-budget", type=int, default=3000)
    parser.add_argument("--output", help="Destination JSONL (must not already exist)")
    args = parser.parse_args()
    if args.limit < 0 or args.token_budget < 1:
        parser.error("limit must be >= 0 and token-budget must be positive")
    data = KGLoader(args.data_path).load()
    supplement = json.loads(Path(args.ontology).read_text(encoding="utf-8")) if args.ontology else None
    ontology = Ontology(data, supplement)
    print(f"Ontology: {len(ontology.parents)} subclass sources, {len(ontology.disjoint)} disjoint axioms. Missing axioms remain unknown.")
    embeddings = Embeddings(args.embeddings, data)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    pipeline = EvidencePipeline(data, embeddings, ontology, args.token_budget, tokenizer)
    if args.queries:
        with Path(args.queries).open(encoding="utf-8") as stream:
            queries = [(int(x["query_head_id"]), int(x["query_relation_id"]))
                       for line in stream if line.strip() for x in [json.loads(line)]]
    else:
        queries = [(q.h_id, q.r_id) for q in getattr(data, args.split + "_triples")]
    if args.limit:
        queries = queries[:args.limit]
    output = Path(args.output or f"output/evidence/{data.dataset_name}/{args.split}_evidence.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for i, (head, relation) in enumerate(queries, 1):
            stream.write(json.dumps(pipeline.run_query(head, relation), ensure_ascii=False, allow_nan=False) + "\n")
            if i % 50 == 0:
                stream.flush()
                print(f"Extracted {i}/{len(queries)} queries")
    print(f"Saved {len(queries)} queries to {output}")


if __name__ == "__main__":
    main()
