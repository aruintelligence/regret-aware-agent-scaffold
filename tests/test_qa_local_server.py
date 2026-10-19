"""Authenticated local QA API tests with a fake engine and no external requests."""
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from grounding_contract import ground_answer, verify_grounding
from qa_local_server import make_server
from sovereign_source_registry import grant_source, revoke_source
from pathlib import Path


class FakeEngine:
    model_id = "fake-qa"
    revision = "test-revision"

    def __init__(self):
        self.calls = []
        self.text = "Oregon"

    def answer(self, context, question, *, max_answer_tokens=30):
        self.calls.append((context, question, max_answer_tokens))
        proof = ground_answer(self.text, context, model_revision=self.revision)
        return {
            "accepted": proof["accepted"], "answer": proof["answer"],
            "proof": proof if proof["accepted"] else None,
            "reason": "verified" if proof["accepted"] else "abstained",
        }


class QALocalServerTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.key = b"local-research-api-key-for-tests-32"
        self.server = make_server(self.engine, self.key, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def post(self, payload, *, auth=True, consent=True, owner=None):
        headers = {"Content-Type": "application/json"}
        if owner is not None:
            headers["X-ARU-Owner"] = owner
        if auth:
            headers["Authorization"] = "Bearer " + self.key.decode()
        if consent:
            headers["X-ARU-Consent"] = "research-opt-in"
        request = urllib.request.Request(
            self.url + "/v1/qa/proof",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_authorization_consent_and_proof(self):
        payload = {
            "model": self.engine.model_id,
            "context": "The observatory is in Oregon.",
            "question": "Where is the observatory?",
        }
        self.assertEqual(self.post(payload, auth=False)[0], 401)
        self.assertEqual(self.post(payload, consent=False)[0], 403)
        self.assertEqual(len(self.engine.calls), 0)
        status, body = self.post(payload)
        self.assertEqual(status, 200)
        self.assertTrue(body["accepted"])
        self.assertEqual(body["answer"], "Oregon")
        self.assertTrue(verify_grounding(body["proof"], payload["context"]))
        self.assertFalse(body["automatic_rollout_permitted"])

    def test_abstention_and_schema_rejection(self):
        self.engine.text = "Washington"
        payload = {"model": self.engine.model_id, "context": "Oregon", "question": "Where?"}
        status, body = self.post(payload)
        self.assertEqual(status, 200)
        self.assertFalse(body["accepted"])
        self.assertEqual(body["answer"], "")
        self.assertIsNone(body["proof"])
        self.assertEqual(self.post({**payload, "unexpected_field": 1})[0], 422)
        self.assertEqual(self.post({**payload, "max_answer_tokens": 0})[0], 422)

    def test_consent_registry_enforces_grant_expiry_and_revocation(self):
        source = "The observatory is in Oregon."
        owner = "local-user-opaque-1"
        registry_key = b"separate-consent-registry-test-secret-32"
        with tempfile.TemporaryDirectory() as folder:
            db = str(Path(folder) / "sources.db")
            server = make_server(
                self.engine, self.key, port=0,
                registry_db=db, registry_key=registry_key,
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            original_url = self.url
            self.url = f"http://127.0.0.1:{server.server_port}"
            try:
                payload = {
                    "model": self.engine.model_id,
                    "context": source,
                    "question": "Where is the observatory?",
                }
                self.assertEqual(self.post(payload, owner=owner)[0], 403)
                self.assertEqual(len(self.engine.calls), 0)
                grant = grant_source(
                    db, source, owner=owner, purpose="qa_source",
                    ttl_seconds=300, key=registry_key, consent_asserted=True,
                )
                status, body = self.post(payload, owner=owner)
                self.assertEqual(status, 200)
                self.assertTrue(body["accepted"])
                self.assertEqual(body["source_authorization"]["grant_id"], grant["grant_id"])
                self.assertEqual(self.post(payload, owner="different-owner")[0], 403)
                self.assertEqual(self.post(
                    {**payload, "context": source + " "}, owner=owner,
                )[0], 403)
                revoke_source(
                    db, grant["grant_id"], owner=owner, key=registry_key,
                )
                status, body = self.post(payload, owner=owner)
                self.assertEqual(status, 403)
                self.assertEqual(body["reason"], "grant_revoked_or_expired")
                self.assertEqual(len(self.engine.calls), 1)
            finally:
                self.url = original_url
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_loopback_only(self):
        with self.assertRaises(ValueError):
            make_server(self.engine, self.key, host="0.0.0.0", port=0)


if __name__ == "__main__":
    unittest.main()
