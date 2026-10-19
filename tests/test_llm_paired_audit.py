"""Matched real-model audit contract tests with a fake HTTP model (no model weights)."""
import json
import tempfile
import threading
import unittest
from pathlib import Path

from hf_local_server import make_server
from llm_paired_audit import run_paired_audit


class PromptSensitiveFakeEngine:
    model_id = "paired-audit-fake"
    revision = "fake-only-not-real-weights"

    def __init__(self):
        self.calls = []

    def generate(self, messages, *, max_tokens=96):
        self.calls.append(messages)
        system = messages[0]["content"]
        answer = "Oregon" if "extractive" in system.lower() else "California"
        return {
            "choices": [{"message": {"role": "assistant", "content": answer}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 2},
        }


def fixture(path, count=20):
    context = "A scientist discovered a comet in 1999. The telescope was in Oregon."
    qas = [
        {"id": f"paired-fixture-{i}", "question": "Where was the telescope?",
         "answers": [{"text": "Oregon", "answer_start": context.index("Oregon")}]}
        for i in range(count)
    ]
    Path(path).write_text(json.dumps({
        "version": "1.1", "data": [
            {"title": "fixture", "paragraphs": [{"context": context, "qas": qas}]}
        ],
    }), encoding="utf-8")


class PairedAuditTests(unittest.TestCase):
    def setUp(self):
        self.engine = PromptSensitiveFakeEngine()
        self.server = make_server(self.engine, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = "http://127.0.0.1:" + str(self.server.server_port) + "/v1/chat/completions"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def test_same_questions_both_prompts_independent_human_gold(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "dev.json"
            fixture(dataset)
            out = Path(folder) / "paired"
            report = run_paired_audit(
                str(dataset), str(out), model=self.engine.model_id,
                endpoint=self.endpoint, acknowledge_model_calls=True,
                limit=8, offset=4, replicates=100,
            )
            self.assertEqual(report["n_requested_calls"], 16)
            self.assertEqual(report["n_completed_calls"], 16)
            self.assertEqual(report["n_complete_pairs"], 8)
            self.assertEqual(report["n_failed_calls"], 0)
            self.assertEqual(report["direct"]["baseline_mean_f1"], 0)
            self.assertEqual(report["direct"]["candidate_mean_f1"], 1)
            self.assertEqual(report["direct"]["f1_difference"], 1)
            self.assertEqual(report["article_cluster_interval"]["lower_95"], 1)
            self.assertEqual(report["article_cluster_interval"]["upper_95"], 1)
            self.assertFalse(report["automatic_rollout_permitted"])
            self.assertEqual(len(self.engine.calls), 16)
            self.assertTrue((out / "paired-index.html").exists())
            self.assertIn("Two prompts", (out / "paired-index.html").read_text())
            self.assertNotIn("Where was the telescope", (out / "paired-report.json").read_text())
            self.assertNotIn("answer_start", json.dumps(self.engine.calls))

    def test_consent_and_nonoverwriting(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "dev.json"
            fixture(dataset)
            out = Path(folder) / "paired"
            with self.assertRaises(ValueError):
                run_paired_audit(
                    str(dataset), str(out), model=self.engine.model_id,
                    endpoint=self.endpoint, limit=4, offset=4,
                )
            self.assertEqual(len(self.engine.calls), 0)
            run_paired_audit(
                str(dataset), str(out), model=self.engine.model_id,
                endpoint=self.endpoint, acknowledge_model_calls=True,
                limit=4, offset=4, replicates=100,
            )
            with self.assertRaises(FileExistsError):
                run_paired_audit(
                    str(dataset), str(out), model=self.engine.model_id,
                    endpoint=self.endpoint, acknowledge_model_calls=True,
                    limit=4, offset=4, replicates=100,
                )

    def test_nonoverlapping_followup_subset(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "dev.json"
            fixture(dataset)
            from squad_sentence_retrieval import load_questions
            discovery, _, _ = load_questions(str(dataset), limit=5, offset=0)
            followup, _, _ = load_questions(str(dataset), limit=5, offset=5)
            self.assertFalse({row[0] for row in discovery} & {row[0] for row in followup})
            with self.assertRaises(ValueError):
                load_questions(str(dataset), limit=5, offset=-1)


if __name__ == "__main__":
    unittest.main()
