import copy
import tempfile
import unittest
from pathlib import Path

from selective_risk_dashboard import render_risk_dashboard, save_risk_dashboard
from selective_risk_gate import fit_selective_risk_policy, sign_risk_policy

KEY = b"separate-risk-dashboard-key-32-bytes"


def signed():
    rows = [
        {"sample_id": str(i), "score": 0.9 if i < 150 else 0.1,
         "correct": i < 150, "group": "technical"}
        for i in range(200)
    ]
    policy = fit_selective_risk_policy(
        rows, thresholds=[0.05, 0.5, 0.95], groups=["technical"],
        target_error=0.05, alpha=0.05, min_group_support=30,
        model_revision="model-sha", rubric_id="independent-gold",
        calibration_set_id="calibration-001",
    )
    return sign_risk_policy(policy, key=KEY)


class DashboardTests(unittest.TestCase):
    def test_renders_signed_group_evidence_without_raw_rows(self):
        policy = signed()
        html = render_risk_dashboard(policy, key=KEY)
        self.assertIn("Trust the evidence", html)
        self.assertIn("Every group must qualify", html)
        self.assertIn("HMAC policy signature was verified", html)
        self.assertIn("technical", html)
        self.assertIn("0.5", html)
        self.assertNotIn('"sample_id"', html)
        self.assertIn("Automatic deployment remains disabled", html)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "evidence.html"
            save_risk_dashboard(policy, key=KEY, output_path=target)
            self.assertTrue(target.is_file())
            with self.assertRaises(FileExistsError):
                save_risk_dashboard(policy, key=KEY, output_path=target)

    def test_rejects_modified_policy(self):
        policy = signed()
        changed = copy.deepcopy(policy)
        changed["target_error"] = 0.99
        with self.assertRaises(ValueError):
            render_risk_dashboard(changed, key=KEY)

    def test_escapes_metadata_to_prevent_html_injection(self):
        rows = [
            {"sample_id": str(i), "score": 0.9, "correct": True, "group": "general"}
            for i in range(100)
        ]
        model_name = "model-<img src=x onerror=alert(1)>"
        policy = fit_selective_risk_policy(
            rows, thresholds=[0.5], groups=["general"], target_error=0.1,
            alpha=0.05, min_group_support=30,
            model_revision=model_name, rubric_id="gold", calibration_set_id="id",
        )
        html = render_risk_dashboard(sign_risk_policy(policy, key=KEY), key=KEY)
        self.assertNotIn("<img src=x", html)
        self.assertIn("&lt;img", html)


if __name__ == "__main__":
    unittest.main()
