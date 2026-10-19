import unittest

from sequential_validation import run_sequential_validation


class SequentialValidationTests(unittest.TestCase):
    def test_seeded_null_and_alternative_reproducible(self):
        a = run_sequential_validation(trials=40, tasks=80, seed=11)
        b = run_sequential_validation(trials=40, tasks=80, seed=11)
        self.assertEqual(a, b)
        self.assertFalse(a["real_world_evidence"])
        self.assertEqual(a["conditions"]["null"]["baseline_only_probability"], 0.2)
        self.assertEqual(a["conditions"]["null"]["candidate_only_probability"], 0.2)
        for condition in a["conditions"].values():
            self.assertGreaterEqual(condition["crossing_fraction"], 0)
            self.assertLessEqual(condition["crossing_fraction"], 1)

    def test_invalid_trial_counts_rejected(self):
        with self.assertRaises(ValueError):
            run_sequential_validation(trials=5)
        with self.assertRaises(ValueError):
            run_sequential_validation(tasks=5)


if __name__ == "__main__":
    unittest.main()
