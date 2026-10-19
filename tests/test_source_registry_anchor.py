import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from source_registry_anchor import create_source_checkpoint, verify_source_checkpoint
from sovereign_source_registry import (
    authorize_source, grant_source, revoke_source, verify_registry,
)

KEY = b"separate-registry-hmac-key-test-32-bytes"
ANCHOR = b"another-secret-independent-anchor-32-bytes"
SOURCE = "The telescope is in Oregon."


class SourceRegistryAnchorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.db = str(root / "registry.db")
        self.checkpoint = str(root / "independent-checkpoint.json")
        self.grant = grant_source(
            self.db, SOURCE, owner="opaque-owner", purpose="qa_source",
            ttl_seconds=1000, key=KEY, consent_asserted=True, now=1000,
        )

    def tearDown(self):
        self.temp.cleanup()

    def anchor(self):
        return create_source_checkpoint(
            self.db, self.checkpoint, registry_key=KEY, anchor_key=ANCHOR,
        )

    def check(self):
        return verify_source_checkpoint(
            self.db, self.checkpoint, registry_key=KEY, anchor_key=ANCHOR,
        )

    def test_checkpoint_verifies_and_refuses_overwrite(self):
        self.assertEqual(self.anchor()["events"], 1)
        self.assertTrue(self.check()["verified"])
        with self.assertRaises(FileExistsError):
            self.anchor()

    def test_truncated_revocation_is_detected_even_if_hmac_chain_passes(self):
        revoke_source(
            self.db, self.grant["grant_id"], owner="opaque-owner",
            key=KEY, now=1100,
        )
        self.anchor()
        self.assertTrue(self.check()["verified"])
        with sqlite3.connect(self.db) as db:
            db.execute("DELETE FROM consent_events WHERE seq=2")
        # Original HMAC chain cannot see a removed final revocation.
        self.assertTrue(verify_registry(self.db, key=KEY)["verified"])
        # A historic authorization check could be fooled after rollback.
        self.assertTrue(authorize_source(
            self.db, SOURCE, owner="opaque-owner", purpose="qa_source",
            key=KEY, now=1200,
        )["authorized"])
        # The independently retained checkpoint exposes the rollback.
        verified = self.check()
        self.assertFalse(verified["verified"])
        self.assertTrue(any("truncated" in error for error in verified["errors"]))

    def test_appending_after_checkpoint_preserves_historical_prefix(self):
        self.anchor()
        revoke_source(
            self.db, self.grant["grant_id"], owner="opaque-owner",
            key=KEY, now=1100,
        )
        verified = self.check()
        self.assertTrue(verified["verified"])
        self.assertEqual(verified["anchored_events"], 1)
        self.assertEqual(verified["current_events"], 2)

    def test_wrong_key_or_tampered_checkpoint_is_rejected(self):
        self.anchor()
        wrong = verify_source_checkpoint(
            self.db, self.checkpoint, registry_key=KEY,
            anchor_key=b"incorrect-independent-anchor-key-32",
        )
        self.assertFalse(wrong["verified"])
        path = Path(self.checkpoint)
        record = json.loads(path.read_text())
        record["head_mac"] = "0" * 64
        path.write_text(json.dumps(record))
        self.assertFalse(self.check()["verified"])


if __name__ == "__main__":
    unittest.main()
