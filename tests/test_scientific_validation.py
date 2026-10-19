import math
import unittest

from scientific_validation import analytic_truth, run_validation, simulate_trial


class ScientificValidationTests(unittest.TestCase):
    def test_known_analytic_truth(self):
        truth = analytic_truth()
        self.assertAlmostEqual(truth["baseline"], (0.65 + 0.60 + 0.65) / 3)
        self.assertAlmostEqual(truth["candidate"], (0.80 + 0.70 + 0.78) / 3)
        self.assertAlmostEqual(truth["uplift"], truth["candidate"] - truth["baseline"])

    def test_seeded_trials_are_reproducible(self):
        self.assertEqual(simulate_trial(60, 10), simulate_trial(60, 10))
        self.assertNotEqual(simulate_trial(60, 10), simulate_trial(60, 11))

    def test_falsification_report_distinguishes_simulation_from_real_world(self):
        report = run_validation(trials=3, n=180, seed=5, bootstrap_replicates=100)
        self.assertFalse(report["real_world_validation"])
        self.assertEqual(report["trials"], 3)
        self.assertEqual(report["tasks_per_trial"], 180)
        for name in ("ips", "snips", "crossfit_dr"):
            self.assertTrue(math.isfinite(report["estimator_summary"][name]["mean_uplift"]))
        self.assertLessEqual(report["bootstrap_interval_coverage"]["covered"], 3)


if __name__ == "__main__":
    unittest.main()
