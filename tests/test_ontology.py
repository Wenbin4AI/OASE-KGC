import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from extract_ontology import (LLM, ENTITY_PROMPT, aggregate, clean_types, entity_inputs,
                              load_graph, resolve_axioms, run, select_triples,
                              validate_entities, validate_schema, write_json)


class FakeLLM:
    calls = 0
    hits = 0

    def ask(self, prompt, payload, validate):
        self.calls += 1
        if prompt == ENTITY_PROMPT:
            result = {"entities": [{"id": e["id"], "types": ["Person" if e["id"] == 0 else "City"],
                                    "confidence": 0.95, "reason": "label and context"} for e in payload["entities"]]}
        else:
            result = {"domain": ["Person"], "range": ["City"], "confidence": 0.9,
                      "reason": "observed relation meaning", "axioms": []}
        validate(result)
        return result


class OntologyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        write_json(self.data / "entity.json", [{"value": 0, "label": "Alice", "classname": "LegacyType"},
                   {"value": 1, "label": "Paris"}, {"value": 2, "label": "Rome"}])
        write_json(self.data / "relation.json", [{"id": 0, "label": "livesIn", "range": "Place"}])
        (self.data / "train2id.txt").write_text("2\n0 1 0\n0 2 0\n")
        (self.data / "test2id.txt").write_text("1\n1 2 0\n")
        self.cfg = {"batch_size": 2, "context_per_entity": 12, "min_confidence": .7, "type_aliases": {}}

    def tearDown(self):
        self.temp.cleanup()

    def test_sample_exact_reproducible_and_small(self):
        triples = [(i, 0, i + 1) for i in range(300)]
        self.assertEqual(len(select_triples(triples, "sample", 42)), 200)
        self.assertEqual(select_triples(triples, "sample", 42), select_triples(triples, "sample", 42))
        self.assertEqual(select_triples(triples[:2], "sample", 42), triples[:2])
        self.assertEqual(select_triples(triples, "all", 42), triples)

    def test_train_only_and_no_original_type_leakage(self):
        e, r, triples, hashes = load_graph(self.data, ["train"], "htr")
        self.assertNotIn((1, 0, 2), triples)
        self.assertNotIn("test2id.txt", hashes)
        inputs = entity_inputs(triples[:1], e, r, 12, False)
        self.assertEqual({v["id"] for v in inputs}, {0, 1})
        self.assertTrue(all(not x["existing_types"] for x in inputs))
        self.assertEqual(len(inputs[0]["context"]), 1)

    def test_bad_id_and_header_rejected(self):
        for text in ("1\n0 900 0\n", "3\n0 1 0\n", "0 1\n"):
            (self.data / "train2id.txt").write_text(text)
            with self.assertRaises(ValueError):
                load_graph(self.data, ["train"], "htr")

    def test_deduplication_and_explicit_order(self):
        (self.data / "train2id.txt").write_text("2\n0 0 1\n0 0 1\n")
        self.assertEqual(load_graph(self.data, ["train"], "hrt")[2], [(0, 0, 1)])

    def test_validation_rejects_ids_nan_and_invented_classes(self):
        good = {"id": 0, "types": ["Person"], "confidence": .8, "reason": "label"}
        for rows in ([], [good, good], [dict(good, id=1)], [dict(good, confidence=float("nan"))]):
            with self.assertRaises(ValueError):
                validate_entities({"entities": rows}, [0])
        with self.assertRaises(ValueError):
            validate_schema({"domain": ["Invented"], "range": [], "confidence": .9, "reason": "x", "axioms": []}, ["Person"])

    def test_multi_type_counts_and_unknown(self):
        e, r, triples, _ = load_graph(self.data, ["train"], "htr")
        typed = {0: {"types": ["Person", "Researcher"]}, 1: {"types": ["City"]}, 2: {"types": []}}
        profiles, pairs = aggregate(triples, typed, r)
        self.assertEqual(profiles[0]["domain_counts"]["Person"], 2)
        self.assertEqual(profiles[0]["typed_tail_count"], 1)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(clean_types([" person ", "PERSON", "human"], {"human": "Person"}), ["person"])

    def test_cycles_and_inherited_disjoint_conflict(self):
        def ax(h, rel, t, conf=.9):
            return {"head": h, "relation": rel, "tail": t, "confidence": conf, "reason": "test"}
        accepted, rejected = resolve_axioms([
            ax("Student", "subClassOf", "Person"), ax("Person", "subClassOf", "Student", .8),
            ax("Person", "disjointWith", "Student"), ax("City", "disjointWith", "Person")],
            {0: {"types": ["Student", "City"]}})
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 3)

    def test_full_pipeline_update_and_idempotent_counts(self):
        args = argparse.Namespace(data_path=str(self.data), splits="train", triple_order="htr", mode="all",
                                  seed=2026, output=str(self.root / "out"), use_existing_types=False, dry_run=False)
        with contextlib.redirect_stdout(io.StringIO()):
            first = run(args, self.cfg, FakeLLM())
            second = run(args, self.cfg, FakeLLM())
            self.assertEqual(first["relations"], second["relations"])
            (self.data / "train2id.txt").write_text("1\n0 1 0\n")
            third = run(args, self.cfg, FakeLLM())
        self.assertEqual(third["relations"]["0"]["triple_count"], 1)
        self.assertEqual(third["relations"]["0"]["source_range"], "Place")
        changes = json.loads((self.root / "out/changes.json").read_text())
        self.assertEqual(changes["entities"]["removed"], ["2"])
        self.assertEqual(first["manifest"]["statistics"]["schema_constraints"], 2)

    def test_transport_cache_retry_and_credentials(self):
        cfg = {"base_url": "https://example.invalid/v1", "model": "test", "max_tokens": 500,
               "json_mode": False, "retries": 1, "timeout": 1, "api_key": "secret"}
        client = LLM(cfg, self.root / "cache.db")
        good = {"entities": [{"id": 0, "types": [], "confidence": .1, "reason": "uncertain"}]}
        def response(content):
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())
        try:
            with patch("extract_ontology.request.urlopen", side_effect=[response("broken"), response(json.dumps(good))]) as call, patch("extract_ontology.time.sleep"):
                validate = lambda x: validate_entities(x, [0])
                self.assertEqual(client.ask("system", {"id": 0}, validate), good)
                self.assertEqual(client.ask("system", {"id": 0}, validate), good)
                self.assertEqual(call.call_count, 2)
                self.assertEqual(client.hits, 1)
                self.assertEqual(call.call_args[0][0].full_url, "https://example.invalid/v1/chat/completions")
            with patch("extract_ontology.request.urlopen", side_effect=HTTPError("url", 401, "auth", {}, None)) as call:
                with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
                    client.ask("different", {}, validate)
                self.assertEqual(call.call_count, 1)
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
