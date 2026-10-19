import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from sovereign_source_registry import (
    authorize_source, grant_source, revoke_source, verify_registry,
)

KEY = b"distinct-consent-registry-test-key-32-bytes"
OTHER_KEY = b"wrong-registry-key-not-the-original-32"
SOURCE = "The observatory was in Oregon in 1999."


class SovereignSourceRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.temp.name) / "consent.db")

    def tearDown(self):
        self.temp.cleanup()

    def grant(self, *, now=1000, ttl=100, owner="owner-A", purpose="qa_source"):
        return grant_source(
            self.db, SOURCE, owner=owner, purpose=purpose,
            ttl_seconds=ttl, key=KEY, consent_asserted=True, now=now,
        )

    def authorize(self, *, now=1050, source=SOURCE, owner="owner-A",
                  purpose="qa_source", key=KEY):
        return authorize_source(
            self.db, source, owner=owner, purpose=purpose, key=key, now=now,
        )

    def test_missing_registry_fails_closed(self):
        result = self.authorize()
        self.assertFalse(result["authorized"])
        self.assertEqual(result["reason"], "registry_missing_or_unreadable")

    def test_explicit_consent_required_and_no_plaintext_storage(self):
        with self.assertRaises(PermissionError):
            grant_source(
                self.db, SOURCE, owner="owner-A", purpose="qa_source",
                ttl_seconds=60, key=KEY, now=1000,
            )
        grant = self.grant()
        self.assertTrue(self.authorize()["authorized"])
        self.assertEqual(grant["consent_status"], "operator_attested_not_independently_verified")
        raw = Path(self.db).read_bytes()
        self.assertNotIn(SOURCE.encode("utf-8"), raw)
        self.assertNotIn(b"owner-A", raw)
        self.assertTrue(verify_registry(self.db, key=KEY)["verified"])

    def test_exact_source_owner_purpose_and_expiry(self):
        self.grant()
        self.assertTrue(self.authorize(now=1000)["authorized"])
        self.assertTrue(self.authorize(now=1099.999)["authorized"])
        self.assertFalse(self.authorize(now=999)["authorized"])
        self.assertFalse(self.authorize(now=1100)["authorized"])
        self.assertFalse(self.authorize(source=SOURCE + " ")["authorized"])
        self.assertFalse(self.authorize(owner="owner-B")["authorized"])
        self.assertFalse(self.authorize(purpose="memory_recall")["authorized"])

    def test_revoke_blocks_and_cannot_be_undone_by_wrong_owner(self):
        grant = self.grant()
        with self.assertRaises(PermissionError):
            revoke_source(self.db, grant["grant_id"], owner="owner-B", key=KEY, now=1010)
        revoke_source(self.db, grant["grant_id"], owner="owner-A", key=KEY, now=1010)
        self.assertEqual(self.authorize()["reason"], "grant_revoked_or_expired")
        with self.assertRaises(ValueError):
            revoke_source(self.db, grant["grant_id"], owner="owner-A", key=KEY, now=1020)
        self.assertTrue(verify_registry(self.db, key=KEY)["verified"])

    def test_regrant_after_revocation_requires_new_explicit_grant(self):
        grant = self.grant()
        revoke_source(self.db, grant["grant_id"], owner="owner-A", key=KEY, now=1010)
        self.assertFalse(self.authorize()["authorized"])
        new = self.grant(now=1020)
        self.assertNotEqual(new["grant_id"], grant["grant_id"])
        self.assertEqual(self.authorize()["grant_id"], new["grant_id"])

    def test_hmac_integrity_and_wrong_key_fail_closed(self):
        self.grant()
        self.assertFalse(self.authorize(key=OTHER_KEY)["authorized"])
        with sqlite3.connect(self.db) as conn:
            row = conn.execute("SELECT payload FROM consent_events WHERE seq=1").fetchone()
            payload = json.loads(row[0])
            payload["expires_at"] = 1000000000
            conn.execute(
                "UPDATE consent_events SET payload=? WHERE seq=1",
                (json.dumps(payload),),
            )
        self.assertFalse(self.authorize()["authorized"])
        self.assertEqual(self.authorize()["reason"], "registry_integrity_failed")
        with self.assertRaises(RuntimeError):
            self.grant()

    def test_missing_internal_event_detected(self):
        self.grant()
        self.grant(now=1001)
        with sqlite3.connect(self.db) as conn:
            conn.execute("DELETE FROM consent_events WHERE seq=1")
        self.assertFalse(self.authorize()["authorized"])
        self.assertFalse(verify_registry(self.db, key=KEY)["verified"])

    def test_invalid_grant_and_clock_values_rejected(self):
        for kwargs in (
            {"ttl_seconds": 0}, {"ttl_seconds": 31 * 86400},
            {"ttl_seconds": True}, {"now": float("nan")},
            {"owner": ""}, {"purpose": "\n"},
        ):
            with self.subTest(kwargs=kwargs):
                options = {
                    "owner": "owner-A", "purpose": "qa_source",
                    "ttl_seconds": 60, "key": KEY,
                    "consent_asserted": True, "now": 1000,
                }
                options.update(kwargs)
                with self.assertRaises(ValueError):
                    grant_source(self.db, SOURCE, **options)


if __name__ == "__main__":
    unittest.main()
