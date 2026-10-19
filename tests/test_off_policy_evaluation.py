import unittest
from off_policy_evaluation import Decision, estimate, bootstrap_snips


class OffPolicyTests(unittest.TestCase):
    def test_constant_policy_with_full_overlap(self):
        rows = [Decision("task", "a", 0.5, 1.0),
                Decision("task", "b", 0.5, 0.0)] * 50
        result = estimate(rows, lambda _: "a")
        self.assertAlmostEqual(result["ips"], 1.0)
        self.assertAlmostEqual(result["snips"], 1.0)
        self.assertEqual(result["effective_sample_size"], 50)

    def test_zero_overlap_is_explicit(self):
        result = estimate([Decision("task", "b", 1.0, 0.5)], lambda _: "a")
        self.assertIsNone(result["snips"])
        self.assertIsNotNone(result["warning"])

    def test_invalid_propensity_rejected(self):
        with self.assertRaises(ValueError):
            Decision("task", "a", 0, 1)

    def test_bootstrap_deterministic(self):
        rows = [Decision("task", "a", 0.5, float(i % 2))
                for i in range(100)]
        a = bootstrap_snips(rows, lambda _: "a", seed=9, replicates=100)
        b = bootstrap_snips(rows, lambda _: "a", seed=9, replicates=100)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
