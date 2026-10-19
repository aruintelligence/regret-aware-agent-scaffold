"""Contract tests for local model HTTP bridge; no model download in unit tests."""
import json
import threading
import unittest
import urllib.error
import urllib.request

from hf_local_server import make_server
from llm_executor import run_chat


class FakeEngine:
    model_id = "fake-local-model"
    revision = "test-revision"

    def __init__(self):
        self.calls = []

    def generate(self, messages, *, max_tokens=96):
        self.calls.append((messages, max_tokens))
        return {
            "choices": [{"index": 0, "message": {
                "role": "assistant", "content": "Oregon",
            }}],
            "model": self.model_id,
            "usage": {"prompt_tokens": 20, "completion_tokens": 2},
        }


class HFLocalServerTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.server = make_server(self.engine, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:" + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def _post(self, payload):
        req = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_existing_llm_adapter_works_without_changes(self):
        result = run_chat(
            [{"role": "user", "content": "Where is the telescope?"}],
            model=self.engine.model_id,
            endpoint=self.base + "/v1/chat/completions",
        )
        self.assertEqual(result["text"], "Oregon")
        self.assertEqual(len(self.engine.calls), 1)

    def test_health_exposes_model_provenance_not_prompt(self):
        with urllib.request.urlopen(self.base + "/health", timeout=5) as response:
            health = json.loads(response.read())
        self.assertEqual(health["status"], "ready")
        self.assertEqual(health["model"], self.engine.model_id)
        self.assertEqual(health["revision"], self.engine.revision)

    def test_rejects_unloaded_model_and_streaming(self):
        base = {
            "model": self.engine.model_id,
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0,
        }
        self.assertEqual(self._post({**base, "model": "other"})[0], 422)
        self.assertEqual(self._post({**base, "stream": True})[0], 422)
        self.assertEqual(self._post({**base, "temperature": 0.5})[0], 422)
        self.assertEqual(len(self.engine.calls), 0)

    def test_rejects_external_bind(self):
        with self.assertRaises(ValueError):
            make_server(self.engine, host="0.0.0.0", port=0)


if __name__ == "__main__":
    unittest.main()
