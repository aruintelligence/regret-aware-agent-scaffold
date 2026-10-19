"""Local evidence gateway integration tests; no external network or credentials."""
import json
import sqlite3
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from evidence_gateway import build_report_from_gateway, make_server
from evidence_audit import verify_manifest
from evidence_checkpoint import create_checkpoint
from evidence_gateway_store import (
    EvidenceConflict, export_jsonl, record_event, verify_chain,
)
from route_experiments import action_id, load_decisions


def decision():
    route = ["research_node"]
    return {
        "schema_version": 1, "decision_id": "test-decision-1",
        "timestamp_unix": 0, "task_signature": "public|research",
        "logging_policy": "categorical",
        "candidate_probabilities": [1.0], "propensity": 1.0,
        "action": action_id(route),
        "candidates": [{"route": route, "action": action_id(route), "predicted_score": 0.5}],
        "runtime_kind": "external_instrumented", "loop_count": 0,
    }


def label():
    return {
        "schema_version": 1, "decision_id": "test-decision-1",
        "reward": 0.8, "grader": "external-1", "rubric_id": "rubric-1",
        "source_kind": "external", "graded_at_unix": 0,
    }


class GatewayStoreTests(unittest.TestCase):
    def test_hmac_chain_idempotency_and_export(self):
        with tempfile.TemporaryDirectory() as folder:
            db = str(Path(folder) / "events.db")
            key = b"local-research-hmac-key-32-bytes"
            a = record_event(db, "decision", decision(), hmac_key=key)
            b = record_event(db, "decision", decision(), hmac_key=key)
            self.assertFalse(a["duplicate"])
            self.assertTrue(b["duplicate"])
            self.assertEqual(a["seq"], b["seq"])
            with self.assertRaises(EvidenceConflict):
                record_event(db, "decision", {**decision(), "timestamp_unix": 1},
                             hmac_key=key)
            record_event(db, "label", label(), hmac_key=key)
            result = verify_chain(db, hmac_key=key)
            self.assertTrue(result["verified"])
            self.assertEqual(result["events"], 2)
            self.assertFalse(verify_chain(db, hmac_key=b"wrong-test-hmac-key-32-bytes")["verified"])
            d = str(Path(folder) / "decisions.jsonl")
            l = str(Path(folder) / "labels.jsonl")
            exported = export_jsonl(db, d, l, hmac_key=key)
            self.assertEqual(exported["decisions"], 1)
            self.assertEqual(exported["labels"], 1)
            self.assertEqual(len(load_decisions(d)), 1)
            with self.assertRaises(FileExistsError):
                export_jsonl(db, d, l, hmac_key=key)

    def test_tampering_detected_and_export_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            db = str(Path(folder) / "events.db")
            key = b"local-research-hmac-key-32-bytes"
            record_event(db, "decision", decision(), hmac_key=key)
            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE events SET payload=? WHERE seq=1", ('{"tampered":true}',))
            result = verify_chain(db, hmac_key=key)
            self.assertFalse(result["verified"])
            with self.assertRaises(RuntimeError):
                export_jsonl(db, str(Path(folder) / "d.jsonl"),
                             str(Path(folder) / "l.jsonl"), hmac_key=key)

    def test_gateway_generates_authenticated_dashboard_passport(self):
        with tempfile.TemporaryDirectory() as folder:
            db = str(Path(folder) / "events.db")
            key = b"local-research-hmac-key-32-bytes"
            first = ["research_node", "coherence_node"]
            second = ["coherence_node"]
            for i in range(30):
                row = decision()
                row["decision_id"] = f"report-{i}"
                row["candidate_probabilities"] = [0.5, 0.5]
                row["propensity"] = 0.5
                row["candidates"] = [
                    {"route": first, "action": action_id(first), "predicted_score": 0.9},
                    {"route": second, "action": action_id(second), "predicted_score": 0.7},
                ]
                row["action"] = action_id(first if i % 2 == 0 else second)
                record_event(db, "decision", row, hmac_key=key)
                outcome = label()
                outcome["decision_id"] = row["decision_id"]
                outcome["reward"] = float(i % 2)
                record_event(db, "label", outcome, hmac_key=key)
            anchor_key = b"independent-test-anchor-secret-32-bytes"
            checkpoint = str(Path(folder) / "anchor.json")
            create_checkpoint(db, checkpoint, gateway_key=key, anchor_key=anchor_key)
            out = Path(folder) / "passport"
            result = build_report_from_gateway(
                db, str(out), hmac_key=key,
                baseline_policy="first_candidate",
                candidate_policy="second_candidate", replicates=100,
                anchor_checkpoint=checkpoint, anchor_key=anchor_key,
            )
            self.assertEqual(result["n_labeled"], 30)
            report_json = json.loads((out / "report.json").read_text())
            self.assertTrue(report_json["source"]["independent_checkpoint_verified"])
            self.assertFalse(result["automatic_rollout_permitted"])
            self.assertTrue((out / "index.html").exists())
            self.assertTrue(verify_manifest(
                str(out / "attestation.json"),
                str(out / "decisions.jsonl"),
                str(out / "labels.jsonl"),
                report_path=str(out / "report.json"), secret=key,
            )["verified"])
            # A valid prefix still passes the ordinary HMAC chain after tail deletion,
            # but must never produce a report when an independent anchor is required.
            with sqlite3.connect(db) as conn:
                conn.execute("DELETE FROM events WHERE seq=(SELECT MAX(seq) FROM events)")
            self.assertTrue(verify_chain(db, hmac_key=key)["verified"])
            blocked = Path(folder) / "blocked"
            with self.assertRaises(RuntimeError):
                build_report_from_gateway(
                    db, str(blocked), hmac_key=key,
                    baseline_policy="first_candidate",
                    candidate_policy="second_candidate", replicates=100,
                    anchor_checkpoint=checkpoint, anchor_key=anchor_key,
                )
            self.assertFalse((blocked / "report.json").exists())

    def test_schema_rejects_raw_prompts_and_unlinked_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            db = str(Path(folder) / "events.db")
            key = b"local-research-hmac-key-32-bytes"
            with self.assertRaises(ValueError):
                record_event(db, "decision", {**decision(), "user_input": "secret"}, hmac_key=key)
            with self.assertRaises(ValueError):
                record_event(db, "label", label(), hmac_key=key)
            with self.assertRaises(ValueError):
                record_event(db, "decision", {**decision(), "timestamp_unix": float("nan")},
                             hmac_key=key)
            with self.assertRaises(ValueError):
                record_event(db, "decision", {**decision(), "loop_count": -1},
                             hmac_key=key)


class GatewayHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.key = b"local-test-api-key-32-bytes-long"
        self.hmac_key = b"local-research-hmac-key-32-bytes"
        self.server = make_server(
            str(Path(self.temp.name) / "events.db"),
            self.key, self.hmac_key, port=0,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.temp.cleanup()

    def request(self, path, *, payload=None, auth=True, consent=True):
        headers = {}
        if auth:
            headers["Authorization"] = "Bearer " + self.key.decode()
        if consent:
            headers["X-ARU-Consent"] = "research-opt-in"
        if payload is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers=headers,
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=3) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_consent_auth_idempotency_and_audit_endpoint(self):
        status, _ = self.request("/health", auth=False)
        self.assertEqual(status, 200)
        status, _ = self.request("/v1/decisions", payload=decision(), auth=False)
        self.assertEqual(status, 401)
        status, _ = self.request("/v1/decisions", payload=decision(), consent=False)
        self.assertEqual(status, 403)
        status, row = self.request("/v1/decisions", payload=decision())
        self.assertEqual(status, 201)
        self.assertFalse(row["duplicate"])
        status, row = self.request("/v1/decisions", payload=decision())
        self.assertEqual(status, 200)
        self.assertTrue(row["duplicate"])
        status, _ = self.request("/v1/labels", payload=label())
        self.assertEqual(status, 201)
        status, chain = self.request("/v1/chain")
        self.assertEqual(status, 200)
        self.assertTrue(chain["verified"])
        self.assertEqual(chain["events"], 2)

    def test_concurrent_decisions_preserve_audit_chain(self):
        def send(i):
            row = decision()
            row["decision_id"] = f"concurrent-{i}"
            return self.request("/v1/decisions", payload=row)[0]
        with ThreadPoolExecutor(max_workers=5) as pool:
            statuses = list(pool.map(send, range(12)))
        self.assertEqual(statuses, [201] * 12)
        status, chain = self.request("/v1/chain")
        self.assertEqual(status, 200)
        self.assertTrue(chain["verified"])
        self.assertEqual(chain["events"], 12)

    def test_rejects_unconsented_sensitive_fields(self):
        status, _ = self.request("/v1/decisions", payload={**decision(), "prompt": "private"})
        self.assertEqual(status, 422)

    def test_loopback_only(self):
        with self.assertRaises(ValueError):
            make_server(str(Path(self.temp.name) / "remote.db"),
                        self.key, self.hmac_key, host="0.0.0.0", port=0)


if __name__ == "__main__":
    unittest.main()
