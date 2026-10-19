import unittest

from selective_risk_validation import run_risk_falsification


class RiskFalsificationTests(unittest.TestCase):
    def test_deterministic_simulation(self):
        a = run_risk_falsification(trials=25, rows_per_group=60, seed=13)
        b = run_risk_falsification(trials=25, rows_per_group=60, seed=13)
        self.assertEqual(a, b)
        self.assertFalse(a["real_world_evidence"])
        self.assertFalse(a["automatic_rollout_permitted"])
        for result in a["conditions"].values():
            self.assertGreaterEqual(result["pass_fraction"], 0)
            self.assertLessEqual(result["pass_fraction"], 1)

    def test_invalid_sample_size_rejected(self):
        with self.assertRaises(ValueError):
            run_risk_falsification(trials=10)
        with self.assertRaises(ValueError):
            run_risk_falsification(rows_per_group=10)


if __name__ == "__main__":
    unittest.main()
