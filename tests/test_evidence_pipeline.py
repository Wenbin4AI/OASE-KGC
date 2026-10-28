import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from evidence_pipeline import Ontology, EvidencePipeline, Embeddings, main
from extract_ontology import load_graph


def fixture():
    data = NS(dataset_name="FB15k-237",
              entities={i: NS(label=f"Entity{i}", classname="Untrusted", raw={}) for i in range(4)},
              relations={0: NS(label="query", domain="Wrong", range="Wrong"),
                         1: NS(label="link", domain="Wrong", range="Wrong")},
              train_triples=[NS(h_id=h, r_id=r, t_id=t) for h, r, t in
                             [(0, 0, 3), (0, 1, 1), (1, 1, 2), (2, 1, 0), (3, 1, 0)]])
    onto = {"entities": {"0": {"types": ["Researcher"]}, "1": {"types": ["City"]},
                         "2": {"types": ["Place"]}},
            "relations": {"0": {"domain": ["Person"], "range": ["Place"]},
                          "1": {"domain": ["Person"], "range": ["Place"]}},
            "axioms": [{"head": "Researcher", "relation": "subClassOf", "tail": "Person"},
                       {"head": "City", "relation": "subClassOf", "tail": "Place"},
                       {"head": "City", "relation": "disjointWith", "tail": "Person"}],
            "manifest": {"mode": "sample", "splits": ["train"]}}
    emb = NS(entity=np.array([[1, 1], [1, 1j], [1j, 1], [-1, 1]], dtype=complex),
             relation=np.array([[1, 1], [1j, 1]], dtype=complex))
    return data, emb, Ontology(data, onto)


class EvidenceTests(unittest.TestCase):
    def test_generated_ontology_replaces_legacy_and_missing_unknown(self):
        data, emb, ontology = fixture()
        self.assertNotIn("untrusted", ontology.classes[0])
        self.assertEqual(ontology.classes[3], frozenset())
        self.assertEqual(ontology.atomic(3, "person"), (.6, "unknown"))
        self.assertEqual(ontology.atomic(0, "person")[0], 1)
        self.assertEqual(ontology.atomic(2, "city")[0], .9)
        self.assertEqual(ontology.atomic(1, "person")[0], .1)

    def test_conjunction_and_positive_priority(self):
        _, _, ontology = fixture()
        self.assertAlmostEqual(ontology.compatibility(0, ["Person", "City"])[0], .1)
        ontology.classes[3] = frozenset(["person", "city"])
        self.assertEqual(ontology.atomic(3, "person")[0], 1)

    def test_supported_overlap(self):
        data, _, _ = fixture()
        ontology = Ontology(data, {"entities": {"0": {"types": ["Teacher"]},
                                               "1": {"types": ["TeachingResearcher"]}},
            "relations": {}, "axioms": [
                {"head": "TeachingResearcher", "relation": "subClassOf", "tail": "Teacher"},
                {"head": "TeachingResearcher", "relation": "subClassOf", "tail": "Researcher"}]})
        self.assertEqual(ontology.atomic(0, "researcher"), (.65, "supported_overlap"))

    def test_equations_2_3_and_5(self):
        data, emb, ontology = fixture()
        p = EvidencePipeline(data, emb, ontology)
        path = ((0, 1, 1), (1, 1, 2))
        score, parts = p.structural(path, 0, 0)
        relevance = (1 + p.cosine(emb.relation[1] ** 2, emb.relation[0])) / 2
        expected = math.exp(-.5) * (.5 * relevance + .5 * math.exp(-np.linalg.norm(emb.entity[0] - emb.entity[2]) / 1.2))
        self.assertAlmostEqual(score, expected)
        one, _ = p.structural(path[:1], 0, 0)
        expected = .4 * (1 + p.cosine(emb.relation[1], emb.relation[0])) / 2 + .6 * math.exp(-np.linalg.norm(emb.entity[0] * emb.relation[1] - emb.entity[1]) / .8)
        self.assertAlmostEqual(one, expected)
        for item in p.run_query(0, 0)["evidence_pool"]:
            self.assertAlmostEqual(item["quality"], item["structural_score"] * math.prod(ontology.triple(*e)[0] for e in item["triples"]))

    def test_no_query_edges_no_cycle_and_budget(self):
        data, emb, ontology = fixture()
        p = EvidencePipeline(data, emb, ontology, budget=85)
        result = p.run_query(0, 0)
        self.assertLessEqual(result["evidence_cost"], 85)
        for item in result["evidence_pool"]:
            self.assertNotIn((0, 0, 3), item["triples"])
            visited = [item["triples"][0][0]] + [e[2] for e in item["triples"]]
            self.assertEqual(len(visited), len(set(visited)))
            self.assertLessEqual(item["quality"], item["structural_score"])
        p.budget = 1
        self.assertEqual(p.run_query(0, 0)["selected_evidence"], [])

    def test_inverse_graph_and_schema(self):
        data, emb, ontology = fixture()
        p = EvidencePipeline(data, emb, ontology, reverse=True)
        self.assertIn((1, 1, 0), p.train)
        np.testing.assert_array_equal(p.emb.relation, np.conj(emb.relation))
        result = p.run_query(1, 0)
        item = next(x for x in result["evidence_pool"] if x["triples"] == ((1, 1, 0),))
        self.assertEqual(item["schema_checks"][0]["domain"][0], 1)
        self.assertEqual(item["schema_checks"][0]["range"][0], 1)
        self.assertIn("inverse of link", item["text"])

    def test_self_loop_not_extended_into_simple_paths(self):
        data, emb, ontology = fixture()
        data.train_triples.append(NS(h_id=0, r_id=1, t_id=0))
        p = EvidencePipeline(data, emb, ontology)
        for path in p.extract(0, 0):
            if len(path) > 1:
                visited = [path[0][0]] + [e[2] for e in path]
                self.assertEqual(len(visited), len(set(visited)))

    def test_candidate_order_matches_rotate_and_no_answer_field(self):
        data, emb, ontology = fixture()
        p = EvidencePipeline(data, emb, ontology)
        result = p.run_query(0, 0)
        candidates = p.candidates(0, 0, result["evidence_pool"], 2, 1)
        expected = sorted(data.entities, key=lambda i: (np.abs(emb.entity[0] - emb.entity[i]).sum(), i))[:2]
        self.assertEqual([x["entity_id"] for x in candidates[:2]], expected)
        self.assertEqual(len({x["entity_id"] for x in candidates}), len(candidates))
        messages = p.messages(result, candidates)
        self.assertNotIn("gold", json.dumps(messages))
        self.assertEqual(json.loads(messages[1]["content"])["query"]["predict"], "tail")

    def test_singleton_comparison(self):
        data, emb, ontology = fixture()
        p = EvidencePipeline(data, emb, ontology, budget=10)
        pool = [{"quality": .8, "cost": 10, "triples": ((0, 1, 1),)},
                {"quality": .1, "cost": 1, "triples": ((0, 1, 1),)}]
        selected, objective = p.select(pool)
        self.assertEqual(selected, [pool[0]])
        self.assertAlmostEqual(objective, .5 * .8 / 2 + .5 * .8)

    def test_cli_end_to_end_and_provenance_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, value in {"entity.json": [{"id": i, "label": f"Entity{i}"} for i in range(4)],
                                "relation.json": [{"id": i, "label": f"r{i}"} for i in range(2)]}.items():
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            (root / "train2id.txt").write_text("2\n0 1 1\n1 2 1\n")
            (root / "test2id.txt").write_text("2\n0 3 0\n0 2 0\n")
            _, _, _, hashes = load_graph(root, ["train"], "htr")
            ontology = {"entities": {}, "relations": {}, "axioms": [],
                        "manifest": {"splits": ["train"], "source_hashes": hashes, "mode": "all"}}
            (root / "ontology.json").write_text(json.dumps(ontology))
            (root / "config.json").write_text('{"gamma":12,"hidden_dim":2}')
            np.save(root / "entity_embedding.npy", np.ones((4, 4)))
            np.save(root / "relation_embedding.npy", np.zeros((2, 2)))
            output = root / "out.jsonl"
            argv = ["evidence_pipeline.py", "--data-path", str(root), "--embeddings", str(root),
                    "--ontology", str(root / "ontology.json"), "--prediction", "both", "--output", str(output)]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            rows = [json.loads(x) for x in output.read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["evidence_text"], rows[2]["evidence_text"])
            self.assertEqual(rows[0]["llm_messages"], rows[2]["llm_messages"])
            self.assertEqual(rows[1]["query_tail_id"], 3)
            self.assertNotIn("query_head_id", rows[1])
            (root / "train2id.txt").write_text("1\n0 1 1\n")
            data, _, _ = fixture()
            with self.assertRaisesRegex(ValueError, "mismatch"):
                Ontology(data, ontology).validate_provenance(root)


if __name__ == "__main__":
    unittest.main()
