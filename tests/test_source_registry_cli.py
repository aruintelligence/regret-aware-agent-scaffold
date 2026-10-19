import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SourceRegistryCLITests(unittest.TestCase):
    def test_operator_consent_grant_check_revoke(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            db = root / "registry.db"
            source = root / "source.txt"
            owner = root / "owner.txt"
            source.write_text("The telescope is in Oregon.", encoding="utf-8")
            owner.write_text("opaque-owner-1\n", encoding="utf-8")
            script = str(Path(__file__).resolve().parents[1] / "source_registry_cli.py")
            env = {
                **os.environ,
                "ARU_SOURCE_REGISTRY_KEY": "independent-test-source-registry-secret-32",
                "ARU_SOURCE_ANCHOR_KEY": "another-independent-anchor-secret-32-bytes",
            }
            def call(*args):
                return subprocess.run(
                    [sys.executable, script, *args],
                    capture_output=True, text=True, env=env,
                )
            basic = ["--db", str(db), "--source-file", str(source),
                     "--owner-file", str(owner)]
            denied = call("check", *basic)
            self.assertEqual(denied.returncode, 1)
            no_consent = call("grant", *basic)
            self.assertNotEqual(no_consent.returncode, 0)
            granted = call("grant", *basic, "--consent-asserted", "--ttl-seconds", "120")
            self.assertEqual(granted.returncode, 0, granted.stderr)
            grant_id = json.loads(granted.stdout)["grant_id"]
            self.assertEqual(call("check", *basic).returncode, 0)
            self.assertEqual(call("verify", "--db", str(db)).returncode, 0)
            revoked = call("revoke", "--db", str(db), "--grant-id", grant_id,
                           "--owner-file", str(owner))
            self.assertEqual(revoked.returncode, 0, revoked.stderr)
            self.assertEqual(call("check", *basic).returncode, 1)
            checkpoint = root / "revocation-checkpoint.json"
            anchored = call(
                "anchor", "--db", str(db), "--checkpoint", str(checkpoint),
            )
            self.assertEqual(anchored.returncode, 0, anchored.stderr)
            self.assertEqual(call(
                "verify-anchor", "--db", str(db), "--checkpoint", str(checkpoint),
            ).returncode, 0)
            self.assertNotIn("The telescope", db.read_bytes().decode("utf-8", errors="ignore"))


if __name__ == "__main__":
    unittest.main()
