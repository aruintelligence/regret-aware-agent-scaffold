"""Real-model adapter tests use a local mock server, never a paid API."""
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from llm_executor import EndpointError, run_chat, validate_endpoint
from llm_squad_benchmark import (
    build_messages, run_llm_benchmark, score_response, squad_f1,
)


class MockHandler(BaseHTTPRequestHandler):
    requests = []
    mode = "success"

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(size))
        type(self).requests.append(payload)
        if type(self).mode == "redirect":
            self.send_response(302)
            self.send_header("Location", "https://example.com/v1/chat/completions")
            self.end_headers()
            return
        if type(self).mode == "error":
            self.send_response(500)
            self.end_headers()
            return
        body = json.dumps({
            "choices": [{"message": {"content": "Oregon"}}],
            "usage": {"prompt_tokens": 25, "completion_tokens": 2},
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def fixture(path, count=12):
    context = "The observatory was built in Oregon. Its director discovered a comet."
    qas = [{"id": f"llm-fixture-{i}", "question": "Where was the observatory built?",
            "answers": [{"text": "Oregon", "answer_start": context.index("Oregon")}]}
           for i in range(count)]
    Path(path).write_text(json.dumps({"version": "1.1", "data": [
        {"title": "fixture", "paragraphs": [{"context": context, "qas": qas}]}
    ]}), encoding="utf-8")


class RealLLMAdapterTests(unittest.TestCase):
    def setUp(self):
        MockHandler.requests = []
        MockHandler.mode = "success"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def test_reject_remote_without_explicit_https_opt_in(self):
        with self.assertRaises(ValueError):
            validate_endpoint("https://api.example.com/v1/chat/completions")
        with self.assertRaises(ValueError):
            validate_endpoint("http://api.example.com/v1/chat/completions", allow_remote=True)
        self.assertEqual(validate_endpoint(self.endpoint), self.endpoint)

    def test_real_http_contract_and_no_annotation_leak(self):
        messages = build_messages("The telescope is in Oregon.", "Where is the telescope?", 1)
        result = run_chat(messages, model="mock-local", endpoint=self.endpoint)
        self.assertEqual(result["text"], "Oregon")
        self.assertEqual(result["prompt_tokens"], 25)
        self.assertEqual(len(MockHandler.requests), 1)
        self.assertNotIn("answer_start", json.dumps(MockHandler.requests[0]))

    def test_redirects_cannot_escape_local_endpoint(self):
        MockHandler.mode = "redirect"
        with self.assertRaises(EndpointError):
            run_chat([{"role": "user", "content": "hello"}],
                     model="mock", endpoint=self.endpoint)

    def test_human_gold_token_f1(self):
        self.assertEqual(squad_f1("The Oregon", "Oregon"), 1)
        self.assertEqual(score_response("Oregon", [{"text": "Oregon"}])["exact_match"], 1)
        self.assertEqual(squad_f1("California", "Oregon"), 0)

    def test_explicit_acknowledgement_required(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "dev.json"
            fixture(dataset)
            with self.assertRaises(ValueError):
                run_llm_benchmark(str(dataset), str(Path(folder) / "out"),
                                  model="mock", endpoint=self.endpoint)
            self.assertEqual(len(MockHandler.requests), 0)

    def test_full_model_integration_with_mock_server(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "dev.json"
            fixture(dataset)
            out = Path(folder) / "out"
            report = run_llm_benchmark(
                str(dataset), str(out), model="mock-local",
                endpoint=self.endpoint, acknowledge_model_calls=True,
                limit=12, seed=17, replicates=100,
            )
            self.assertEqual(report["llm_benchmark"]["n_completed"], 12)
            self.assertEqual(report["llm_benchmark"]["n_failed"], 0)
            self.assertEqual(report["label_source_counts"]["external"], 12)
            self.assertFalse(report["review_gate"]["automatic_rollout_permitted"])
            self.assertEqual(len(MockHandler.requests), 12)
            self.assertTrue((out / "index.html").exists())
            self.assertNotIn("Where was the observatory built?", (out / "decisions.llm.jsonl").read_text())

    def test_failed_requests_never_fabricate_outcomes(self):
        MockHandler.mode = "error"
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "dev.json"
            fixture(dataset, count=3)
            out = Path(folder) / "out"
            with self.assertRaises(RuntimeError):
                run_llm_benchmark(
                    str(dataset), str(out), model="mock-local",
                    endpoint=self.endpoint, acknowledge_model_calls=True,
                    limit=3, seed=17, replicates=100,
                )
            self.assertEqual(len((out / "decisions.llm.jsonl").read_text().splitlines()), 3)
            self.assertFalse((out / "labels.llm.jsonl").exists())
            self.assertEqual(len((out / "failures.llm.jsonl").read_text().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
