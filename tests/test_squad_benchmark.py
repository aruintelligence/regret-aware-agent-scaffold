"""Real-task benchmark tests use tiny hand-authored fixtures, not SQuAD copies."""
import json
import tempfile
import unittest
from pathlib import Path

from squad_benchmark import paired_oracle_interval, run_benchmark
from squad_sentence_retrieval import (
    gold_span_contained, idf_proximity_sentence, load_questions,
    overlap_sentence, segment_context,
)


def write_fixture(path, count=24):
    context = "A scientist discovered a comet in 1999. The telescope was in Oregon. The team published the discovery."
    start = context.index("Oregon")
    qas = [
        {"id": f"fixture-question-{i}", "question": "Where was the telescope?",
         "answers": [{"text": "Oregon", "answer_start": start}]}
        for i in range(count)
    ]
    payload = {"version": "1.1", "data": [
        {"title": "fixture", "paragraphs": [{"context": context, "qas": qas}]}
    ]}
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


class SquadSentenceTests(unittest.TestCase):
    def test_segmentation_preserves_offsets(self):
        text = "Alpha arrived. Beta stayed. Gamma left."
        spans = segment_context(text)
        self.assertEqual(len(spans), 3)
        for start, end, sentence in spans:
            self.assertEqual(text[start:end], sentence)

    def test_algorithms_only_take_question_and_context(self):
        context = "A comet was found in 1999. The telescope was in Oregon."
        question = "Where was the telescope?"
        baseline = overlap_sentence(question, context)
        candidate = idf_proximity_sentence(question, context)
        answers = [{"text": "Oregon", "answer_start": context.index("Oregon")}]
        self.assertTrue(gold_span_contained(baseline, answers))
        self.assertTrue(gold_span_contained(candidate, answers))

    def test_gold_offsets_must_match_source_text(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "dev.json"
            write_fixture(dataset)
            questions, sha, total = load_questions(str(dataset), limit=5)
            self.assertEqual(len(questions), 5)
            self.assertEqual(total, 24)
            payload = json.loads(dataset.read_text())
            payload["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["answer_start"] = 0
            dataset.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_questions(str(dataset))

    def test_oracle_interval_uses_paired_questions(self):
        rows = [(1, 0)] * 12
        interval = paired_oracle_interval(rows, replicates=100, seed=4)
        self.assertEqual(interval["lower_95"], -1)
        self.assertEqual(interval["upper_95"], -1)

    def test_benchmark_generates_real_gold_labels_without_llm_claims(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "dev.json"
            write_fixture(dataset)
            out = Path(folder) / "report"
            report = run_benchmark(str(dataset), str(out), limit=24,
                                   seed=7, replicates=100)
            self.assertEqual(report["n_labeled"], 24)
            self.assertEqual(report["label_source_counts"]["external"], 24)
            self.assertEqual(report["external_benchmark"]["n_evaluated"], 24)
            self.assertFalse(report["review_gate"]["automatic_rollout_permitted"])
            self.assertTrue((out / "index.html").exists())
            self.assertEqual(len((out / "decisions.benchmark.jsonl").read_text().splitlines()), 24)
            self.assertEqual(len((out / "labels.benchmark.jsonl").read_text().splitlines()), 24)
            self.assertIn("not LLM", report["external_benchmark"]["scope"])
            with self.assertRaises(FileExistsError):
                run_benchmark(str(dataset), str(out), limit=24,
                              seed=7, replicates=100)


if __name__ == "__main__":
    unittest.main()
