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
        self.answer = "Oregon"

    def generate(self, messages, *, max_tokens=96):
        self.calls.append((messages, max_tokens))
        return {
            "choices": [{"index": 0, "message": {
                "role": "assistant", "content": self.answer,
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

    def _post(self, payload, *, path="/v1/chat/completions", consent=False):
        headers = {"Content-Type": "application/json"}
        if consent:
            headers["X-ARU-Consent"] = "research-opt-in"
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
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

    def test_grounded_answer_api_requires_consent_and_proves_source_span(self):
        payload = {
            "model": self.engine.model_id,
            "context": "The observatory was in Oregon.",
            "question": "Where was the observatory?",
            "max_tokens": 48,
        }
        path = "/v1/grounded/answers"
        status, _ = self._post(payload, path=path)
        self.assertEqual(status, 403)
        self.assertEqual(len(self.engine.calls), 0)
        status, body = self._post(payload, path=path, consent=True)
        self.assertEqual(status, 200)
        self.assertTrue(body["accepted"])
        self.assertEqual(body["answer"], "Oregon")
        self.assertEqual(body["proof"]["model_revision"], self.engine.revision)
        from grounding_contract import verify_grounding
        self.assertTrue(verify_grounding(body["proof"], payload["context"]))
        self.assertFalse(body["automatic_rollout_permitted"])
        self.assertNotIn(payload["context"], json.dumps(body))

    def test_grounded_answer_api_abstains_on_unsupported_model_claim(self):
        self.engine.answer = "Washington"
        status, body = self._post({
            "model": self.engine.model_id,
            "context": "The observatory was in Oregon.",
            "question": "Where was the observatory?",
        }, path="/v1/grounded/answers", consent=True)
        self.assertEqual(status, 200)
        self.assertFalse(body["accepted"])
        self.assertEqual(body["answer"], "")
        self.assertEqual(body["proof"]["reason"], "model_answer_not_found_as_source_span")

    def test_grounded_api_rejects_unknown_fields_and_oversized_question(self):
        payload = {
            "model": self.engine.model_id,
            "context": "Oregon",
            "question": "Where?",
            "raw_private_data": "secret",
        }
        status, _ = self._post(payload, path="/v1/grounded/answers", consent=True)
        self.assertEqual(status, 422)
        payload.pop("raw_private_data")
        payload["question"] = "A" * 2001
        status, _ = self._post(payload, path="/v1/grounded/answers", consent=True)
        self.assertEqual(status, 422)
        self.assertEqual(len(self.engine.calls), 0)

    def test_rejects_external_bind(self):
        with self.assertRaises(ValueError):
            make_server(self.engine, host="0.0.0.0", port=0)


if __name__ == "__main__":
    unittest.main()
