import unittest

from distribution_robustness import topic_mix_stress_table, worst_case_topic_mix


class DistributionRobustnessTests(unittest.TestCase):
    def test_two_domain_exact_transport(self):
        result = worst_case_topic_mix([0.0, 1.0], [0.5, 0.5], tv_budget=0.1)
        self.assertAlmostEqual(result["observed_uplift"], 0.5)
        self.assertAlmostEqual(result["worst_case_uplift"], 0.4)
        self.assertAlmostEqual(result["realized_tv"], 0.1)

    def test_transport_stops_when_high_value_mass_exhausted(self):
        result = worst_case_topic_mix([0.0, 1.0], [0.8, 0.2], tv_budget=0.5)
        self.assertAlmostEqual(result["worst_case_uplift"], 0)
        self.assertAlmostEqual(result["realized_tv"], 0.2)

    def test_zero_shift_and_single_domain(self):
        result = worst_case_topic_mix([-0.1, 0.2], [0.4, 0.6], tv_budget=0)
        self.assertAlmostEqual(result["worst_case_uplift"], result["observed_uplift"])
        result = worst_case_topic_mix([0.3], [1.0], tv_budget=0.3)
        self.assertAlmostEqual(result["worst_case_uplift"], 0.3)

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            worst_case_topic_mix([0.2], [0.9])
        with self.assertRaises(ValueError):
            worst_case_topic_mix([0.2, 0.4], [0.5, 0.5], tv_budget=1.1)
        with self.assertRaises(ValueError):
            worst_case_topic_mix([float("nan")], [1.0])

    def test_table_reproducible_and_monotone(self):
        diagnostics = [
            {"article": "a", "n": 40, "difference": 0.3},
            {"article": "b", "n": 60, "difference": -0.1},
        ]
        table = topic_mix_stress_table(diagnostics)
        self.assertEqual(len(table), 3)
        self.assertGreaterEqual(table[0]["worst_case_uplift"], table[1]["worst_case_uplift"])
        self.assertGreaterEqual(table[1]["worst_case_uplift"], table[2]["worst_case_uplift"])


if __name__ == "__main__":
    unittest.main()
