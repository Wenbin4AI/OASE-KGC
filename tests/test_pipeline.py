import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline import export_bundle, lines, main, validate_bundle
from evaluate_sample import file_hash, metric_group, full_order, ranks
from extract_ontology import write_json


class PipelineTests(unittest.TestCase):
    def fixture(self, root):
        data, run = root / "data", root / "run"
        data.mkdir()
        (run / "per_query").mkdir(parents=True)
        write_json(data / "entity.json", [{"id": i, "label": str(i)} for i in range(3)])
        write_json(data / "relation.json", [{"id": 0, "label": "r"}])
        (data / "train2id.txt").write_text("2\n0 1 0\n0 2 0\n")
        protocol = {"filter_splits": ["train"], "evaluation": "training_sample_diagnostic_not_held_out"}
        source = {"fingerprint": "test", "prediction": "tail", "targets": [[0, 0, 1]], "protocol": protocol,
                  "source_hashes": {n: file_hash(data / n) for n in ["entity.json", "relation.json", "train2id.txt"]}}
        write_json(run / "manifest.json", source)
        candidates = [{"index": i, "entity_id": i, "label": str(i)} for i in (0, 1)]
        payload = {"query": {"known_entity": "0", "relation": "r", "predict": "tail"},
                   "answer_type_requirements": [], "evidence": "Evidence text", "candidates": candidates}
        row = {"fingerprint": "test", "query_index": 0, "prediction": "tail",
               "gold_triple": {"ids": [0, 0, 1], "labels": ["0", "r", "1"]},
               "predicted_triple": {"ids": [0, 0, 1], "labels": ["0", "r", "1"]},
               "correct": True, "raw_rank": 1, "filtered_rank": 1, "gold_in_candidates": True,
               "full_entity_ranking": [1, 0, 2], "candidates": candidates,
               "llm_messages": [{"role": "system", "content": "rank"}, {"role": "user", "content": json.dumps(payload)}],
               "llm": {"ranking": [1, 0], "fallback_used": False, "attempts": [{"response": '{"ranking":[1,0]}'}]},
               "llm_ranked_entity_ids": [1, 0],
               "evidence": {"evidence_text": "Evidence text", "selected_evidence": [], "evidence_pool": []}}
        write_json(run / "per_query/0000_tail.json", row)
        return data, run, row

    def test_roundtrip_replay_resume_and_ablation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data, run, row = self.fixture(root)
            before = file_hash(run / "per_query/0000_tail.json")
            bundle = root / "bundle"
            with contextlib.redirect_stdout(io.StringIO()):
                export_bundle(run, data, bundle)
            self.assertEqual(before, file_hash(run / "per_query/0000_tail.json"))
            self.assertEqual(validate_bundle(bundle)["query_count"], 1)
            self.assertNotIn("gold_triple", next(lines(bundle / "queries.jsonl.gz")))
            cfg = root / "api.json"
            write_json(cfg, {"base_url": "https://example.invalid/v1", "model": "test"})
            argv = ["replay", "--bundle", str(bundle), "--config", str(cfg), "--output", str(root / "replay")]
            with patch.dict(os.environ, {"LLM_API_KEY": "not-a-real-key"}), patch("pipeline.RankingClient.rank", return_value=row["llm"]) as call, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(argv), 0)
                self.assertEqual(main(argv), 0)
                self.assertEqual(call.call_count, 1)
                sent = json.loads(call.call_args.args[0][1]["content"])
                self.assertNotIn("gold_triple", sent)
                self.assertEqual(sent["evidence"], "Evidence text")
                none_args = argv[:-1] + [str(root / "none"), "--evidence-mode", "none"]
                self.assertEqual(main(none_args), 0)
                without = json.loads(call.call_args.args[0][1]["content"])
                self.assertEqual(without["evidence"], "")
                self.assertEqual(without["candidates"], sent["candidates"])
            metrics = json.loads((root / "replay/metrics.json").read_text())
            self.assertEqual(metrics["overall"]["filtered"]["MRR"], 1)

    def test_checksum_and_source_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data, run, _ = self.fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                export_bundle(run, data, root / "bundle")
            (root / "bundle/labels.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_bundle(root / "bundle")
            (data / "train2id.txt").write_text("1\n0 1 0\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                export_bundle(run, data, root / "other")

    def test_shipped_reference_metrics(self):
        bundle = Path(__file__).resolve().parents[1] / "artifacts/fb15k237-sample200"
        manifest = validate_bundle(bundle)
        refs = list(lines(bundle / "reference_predictions.jsonl.gz"))
        self.assertEqual(len(refs), 200)
        self.assertEqual(manifest["query_count"], 200)
        expected = json.loads((bundle / "reference_metrics.json").read_text())["overall"]
        self.assertEqual(metric_group(refs, [1, 3, 10]), expected)
        reference = {r["id"]: r for r in refs}
        labels = {r["id"]: r for r in json.loads((bundle / "labels.json").read_text())}
        for query in lines(bundle / "queries.jsonl.gz"):
            row, label = reference[query["id"]], labels[query["id"]]
            order = full_order(query["remaining_entity_order"], query["candidates"], row["llm"]["ranking"])
            gold = label["gold_triple"]["ids"][2 if query["prediction"] == "tail" else 0]
            self.assertEqual(ranks(order, gold, set(label["known_answers"])), (row["raw_rank"], row["filtered_rank"]))
            self.assertEqual(order[0] == gold, row["correct"])


if __name__ == "__main__":
    unittest.main()
