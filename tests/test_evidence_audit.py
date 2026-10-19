import argparse
import tempfile
import unittest
from pathlib import Path

from evidence_audit import build_manifest, save_manifest, verify_manifest
from experiment_cli import demo


class EvidenceAuditTests(unittest.TestCase):
    def test_authenticated_manifest_detects_modified_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "demo"
            demo(argparse.Namespace(out_dir=str(root), n=30, seed=11))
            decisions = str(root / "decisions.synthetic.jsonl")
            labels = str(root / "labels.synthetic.jsonl")
            report = str(root / "report.json")
            manifest_path = str(root / "attestation.json")
            key = b"test-only-hmac-secret-32-bytes-long"
            manifest = build_manifest(decisions, labels, report_path=report, secret=key)
            save_manifest(manifest, manifest_path)
            valid = verify_manifest(manifest_path, decisions, labels, report_path=report, secret=key)
            self.assertTrue(valid["verified"])
            wrong_key = verify_manifest(manifest_path, decisions, labels, report_path=report,
                                        secret=b"wrong-test-hmac-secret-32-bytes")
            self.assertFalse(wrong_key["verified"])
            with open(labels, "a", encoding="utf-8") as handle:
                handle.write("tampering\n")
            invalid = verify_manifest(manifest_path, decisions, labels, report_path=report, secret=key)
            self.assertFalse(invalid["verified"])
            self.assertIn("File changed: labels", invalid["errors"])
            with self.assertRaises(FileExistsError):
                save_manifest(manifest, manifest_path)

    def test_unsigned_manifest_is_not_presented_as_authenticated(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "demo"
            demo(argparse.Namespace(out_dir=str(root), n=20, seed=3))
            decisions = str(root / "decisions.synthetic.jsonl")
            labels = str(root / "labels.synthetic.jsonl")
            manifest_path = str(root / "attestation.json")
            save_manifest(build_manifest(decisions, labels), manifest_path)
            result = verify_manifest(manifest_path, decisions, labels)
            self.assertTrue(result["verified"])
            self.assertEqual(result["authentication"], "unsigned-checksums")
            self.assertIn("separately trusted", result["warning"])


if __name__ == "__main__":
    unittest.main()
