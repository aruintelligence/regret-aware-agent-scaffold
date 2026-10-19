"""QA evidence comparison contract tests; no real weights or network downloads."""
import json
import tempfile
import threading
import unittest
from pathlib import Path

from grounding_contract import ground_answer
from hf_local_server import make_server
from qa_proof_benchmark import evaluate_qa_models


class FakeQA:
    model_id = "fake-qa"
    revision = "fake-revision-qa"

    def answer(self, context, question):
        proof = ground_answer("Oregon", context, model_revision=self.revision)
        return {
            "accepted": proof["accepted"], "answer": proof["answer"],
            "proof": proof, "model": self.model_id,
            "model_revision": self.revision,
        }


class FakeLLM:
    model_id = "fake-llm"
    revision = "fake-revision-llm"

    def generate(self, messages, *, max_tokens=96):
        return {
            "choices": [{"message": {"role": "assistant", "content": "California"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        }


def fixture(path):
    context = "The telescope was built in Oregon."
    data = {
        "version": "1.1",
        "data": [{"title": "test-article", "paragraphs": [{
            "context": context,
            "qas": [{
                "id": f"qa-comparison-{i}",
                "question": "Where was the telescope built?",
                "answers": [{"text": "Oregon", "answer_start": context.index("Oregon")}],
            } for i in range(20)],
        }]}],
    }
    Path(path).write_text(json.dumps(data), encoding="utf-8")


class QAProofBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.server = make_server(FakeLLM(), port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def test_matched_qa_comparison_and_source_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "dev.json"
            fixture(dataset)
            out = Path(folder) / "report"
            report = evaluate_qa_models(
                str(dataset), str(out),
                qa_model="fake-qa", llm_model="fake-llm",
                llm_endpoint=self.endpoint,
                acknowledge_model_calls=True, limit=8, offset=4,
                replicates=100, engine=FakeQA(),
            )
            self.assertEqual(report["n_complete_pairs"], 8)
            self.assertEqual(report["failures"], {"qa": 0, "llm": 0})
            self.assertEqual(report["arms"]["qa"]["accepted_fraction"], 1)
            self.assertEqual(report["arms"]["qa"]["mean_f1"], 1)
            self.assertEqual(report["arms"]["llm_raw"]["mean_f1"], 0)
            self.assertEqual(report["arms"]["llm_gated"]["accepted_fraction"], 0)
            self.assertEqual(report["direct_qa_minus_raw_llm_f1"], 1)
            self.assertEqual(report["article_cluster_interval"]["lower_95"], 1)
            self.assertFalse(report["automatic_rollout_permitted"])
            self.assertTrue((out / "index.html").exists())
            self.assertIn("Answers with", (out / "index.html").read_text())
            self.assertNotIn("Where was the telescope", (out / "report.json").read_text())
            with self.assertRaises(FileExistsError):
                evaluate_qa_models(
                    str(dataset), str(out),
                    qa_model="fake-qa", llm_model="fake-llm",
                    llm_endpoint=self.endpoint,
                    acknowledge_model_calls=True, limit=8, offset=4,
                    replicates=100, engine=FakeQA(),
                )

    def test_explicit_consent_required(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "dev.json"
            fixture(dataset)
            with self.assertRaises(ValueError):
                evaluate_qa_models(
                    str(dataset), str(Path(folder) / "out"),
                    qa_model="fake-qa", llm_model="fake-llm",
                    llm_endpoint=self.endpoint, limit=8, offset=4,
                    engine=FakeQA(),
                )


if __name__ == "__main__":
    unittest.main()
