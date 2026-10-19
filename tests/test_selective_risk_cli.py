"""End-to-end signed selective risk policy CLI test with no external services."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from grounding_contract import ground_answer


class RiskCLITests(unittest.TestCase):
    def test_fit_verify_check_and_detect_policy_edit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            calibration = root / "calibration.jsonl"
            with calibration.open("w", encoding="utf-8") as handle:
                for group in ("technical", "general"):
                    for i in range(180):
                        row = {
                            "sample_id": f"{group}-{i}", "group": group,
                            "score": 0.9 if i < 150 else 0.1,
                            "correct": i < 150,
                        }
                        handle.write(json.dumps(row) + "\n")
            policy = root / "signed-policy.json"
            source = root / "source.txt"
            proof_file = root / "proof.json"
            source.write_text("The telescope was in Oregon.", encoding="utf-8")
            proof_file.write_text(json.dumps(ground_answer(
                "Oregon", source.read_text(), model_revision="qa-immutable-sha",
            )), encoding="utf-8")
            env = {**os.environ, "ARU_RISK_SIGNING_KEY": "a-test-signing-key-with-at-least-32-bytes"}
            script = str(Path(__file__).resolve().parents[1] / "selective_risk_cli.py")

            def call(*args):
                return subprocess.run(
                    [sys.executable, script, *args],
                    capture_output=True, text=True, env=env,
                )

            result = call(
                "fit", "--calibration-jsonl", str(calibration),
                "--thresholds", "0.05,0.5,0.95",
                "--groups", "technical,general",
                "--target-error", "0.05",
                "--alpha", "0.05",
                "--min-group-support", "30",
                "--model-revision", "qa-immutable-sha",
                "--rubric-id", "human-gold-v1",
                "--calibration-set-id", "heldout-cal-001",
                "--out", str(policy),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)
            self.assertTrue(summary["conditional_risk_screen_passed"])
            self.assertEqual(summary["selected_threshold"], 0.5)
            self.assertEqual(call("verify", "--policy", str(policy)).returncode, 0)
            dashboard = root / "risk-dashboard.html"
            dashboard_result = call(
                "dashboard", "--policy", str(policy), "--out", str(dashboard),
            )
            self.assertEqual(dashboard_result.returncode, 0, dashboard_result.stderr)
            self.assertTrue(dashboard.is_file())
            self.assertIn("Every group must qualify", dashboard.read_text())
            accepted = call(
                "check", "--policy", str(policy),
                "--score", "0.9", "--group", "technical",
                "--source-file", str(source), "--proof-file", str(proof_file),
                "--model-revision", "qa-immutable-sha",
                "--rubric-id", "human-gold-v1",
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertEqual(json.loads(accepted.stdout)["answer"], "Oregon")
            rejected = call(
                "check", "--policy", str(policy),
                "--score", "0.1", "--group", "technical",
                "--source-file", str(source), "--proof-file", str(proof_file),
                "--model-revision", "qa-immutable-sha",
                "--rubric-id", "human-gold-v1",
            )
            self.assertEqual(rejected.returncode, 1)
            self.assertEqual(json.loads(rejected.stdout)["answer"], "")
            # A modified threshold is not accepted without the signing key.
            changed = json.loads(policy.read_text())
            changed["selected_threshold"] = 0.05
            policy.write_text(json.dumps(changed))
            self.assertEqual(call("verify", "--policy", str(policy)).returncode, 1)

    def test_no_signing_key_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            policy = Path(folder) / "policy.json"
            policy.write_text("{}", encoding="utf-8")
            env = dict(os.environ)
            env.pop("ARU_RISK_SIGNING_KEY", None)
            script = str(Path(__file__).resolve().parents[1] / "selective_risk_cli.py")
            result = subprocess.run(
                [sys.executable, script, "verify", "--policy", str(policy)],
                env=env, capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
