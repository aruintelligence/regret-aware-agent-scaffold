"""Separate-key checkpoint tests, including a real truncation attack."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from evidence_checkpoint import create_checkpoint, verify_checkpoint
from evidence_gateway_store import record_event, verify_chain
from route_experiments import action_id

GATEWAY_KEY = b"gateway-test-key-kept-independent-32"
ANCHOR_KEY = b"separate-anchor-test-key-kept-outside-db-32"


def decision(i):
    route = ["research_node"]
    return {
        "schema_version": 1,
        "decision_id": "anchor-event-" + str(i),
        "timestamp_unix": 0,
        "task_signature": "public|research",
        "logging_policy": "categorical",
        "candidate_probabilities": [1.0],
        "propensity": 1.0,
        "action": action_id(route),
        "candidates": [{
            "route": route, "action": action_id(route), "predicted_score": 0.5,
        }],
        "runtime_kind": "external_instrumented",
        "loop_count": 0,
    }


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.temp.name) / "gateway.db")
        self.anchor = str(Path(self.temp.name) / "independent-anchor.json")
        for i in range(4):
            record_event(self.db, "decision", decision(i), hmac_key=GATEWAY_KEY)

    def tearDown(self):
        self.temp.cleanup()

    def test_checkpoint_verifies_and_is_not_overwritten(self):
        result = create_checkpoint(
            self.db, self.anchor, gateway_key=GATEWAY_KEY, anchor_key=ANCHOR_KEY,
        )
        self.assertEqual(result["events"], 4)
        verified = verify_checkpoint(
            self.db, self.anchor, gateway_key=GATEWAY_KEY, anchor_key=ANCHOR_KEY,
        )
        self.assertTrue(verified["verified"])
        with self.assertRaises(FileExistsError):
            create_checkpoint(
                self.db, self.anchor, gateway_key=GATEWAY_KEY, anchor_key=ANCHOR_KEY,
            )

    def test_tail_truncation_invisible_to_unanchored_chain_is_detected(self):
        create_checkpoint(
            self.db, self.anchor, gateway_key=GATEWAY_KEY, anchor_key=ANCHOR_KEY,
        )
        with sqlite3.connect(self.db) as conn:
            conn.execute("DELETE FROM events WHERE seq=4")
        self.assertTrue(verify_chain(self.db, hmac_key=GATEWAY_KEY)["verified"])
        result = verify_checkpoint(
            self.db, self.anchor, gateway_key=GATEWAY_KEY, anchor_key=ANCHOR_KEY,
        )
        self.assertFalse(result["verified"])
        self.assertTrue(any("truncated" in error for error in result["errors"]))

    def test_wrong_anchor_key_and_checkpoint_edit_detected(self):
        create_checkpoint(
            self.db, self.anchor, gateway_key=GATEWAY_KEY, anchor_key=ANCHOR_KEY,
        )
        wrong = verify_checkpoint(
            self.db, self.anchor, gateway_key=GATEWAY_KEY,
            anchor_key=b"wrong-separate-anchor-test-key-32",
        )
        self.assertFalse(wrong["verified"])
        record = json.loads(Path(self.anchor).read_text())
        record["head_hash"] = "0" * 64
        Path(self.anchor).write_text(json.dumps(record))
        modified = verify_checkpoint(
            self.db, self.anchor, gateway_key=GATEWAY_KEY, anchor_key=ANCHOR_KEY,
        )
        self.assertFalse(modified["verified"])

    def test_append_after_anchor_preserves_historical_prefix(self):
        create_checkpoint(
            self.db, self.anchor, gateway_key=GATEWAY_KEY, anchor_key=ANCHOR_KEY,
        )
        record_event(self.db, "decision", decision(4), hmac_key=GATEWAY_KEY)
        verified = verify_checkpoint(
            self.db, self.anchor, gateway_key=GATEWAY_KEY, anchor_key=ANCHOR_KEY,
        )
        self.assertTrue(verified["verified"])
        self.assertEqual(verified["current_events"], 5)
        self.assertEqual(verified["anchored_events"], 4)

    def test_gap_in_sequence_detected_without_checkpoint(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute("DELETE FROM events WHERE seq=2")
        result = verify_chain(self.db, hmac_key=GATEWAY_KEY)
        self.assertFalse(result["verified"])
        self.assertTrue(any("sequence" in error for error in result["errors"]))

    def test_empty_database_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            db = str(Path(folder) / "empty.db")
            anchor = str(Path(folder) / "anchor.json")
            from evidence_gateway_store import _connect
            with _connect(db):
                pass
            created = create_checkpoint(
                db, anchor, gateway_key=GATEWAY_KEY, anchor_key=ANCHOR_KEY,
            )
            self.assertEqual(created["events"], 0)
            self.assertTrue(verify_checkpoint(
                db, anchor, gateway_key=GATEWAY_KEY, anchor_key=ANCHOR_KEY,
            )["verified"])


if __name__ == "__main__":
    unittest.main()
