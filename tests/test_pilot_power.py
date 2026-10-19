import unittest

from pilot_power import paired_binary_pilot_plan


class PilotPowerTests(unittest.TestCase):
    def test_larger_target_uplift_reduces_required_pairs(self):
        small = paired_binary_pilot_plan(target_uplift=0.03)
        large = paired_binary_pilot_plan(target_uplift=0.08)
        self.assertGreater(small["independent_pairs_required"], large["independent_pairs_required"])

    def test_higher_power_increases_sample(self):
        lower = paired_binary_pilot_plan(power=0.8)
        higher = paired_binary_pilot_plan(power=0.95)
        self.assertLess(lower["independent_pairs_required"], higher["independent_pairs_required"])

    def test_cluster_and_missingness_increase_requested_tasks(self):
        simple = paired_binary_pilot_plan()
        clustered = paired_binary_pilot_plan(
            mean_cluster_size=20, intracluster_correlation=0.1,
            missing_outcome_fraction=0.2,
        )
        self.assertGreater(clustered["requested_tasks_rounded_to_clusters"],
                           simple["requested_tasks_rounded_to_clusters"])
        self.assertAlmostEqual(clustered["cluster_design_effect"], 2.9)
        self.assertEqual(clustered["requested_tasks_rounded_to_clusters"] % 20, 0)

    def test_cost_estimate_uses_operator_assumptions(self):
        plan = paired_binary_pilot_plan(
            grade_cost_per_task=0.5, execution_cost_per_task=0.25,
        )
        self.assertAlmostEqual(
            plan["estimated_task_execution_and_grading_cost"],
            plan["requested_tasks_rounded_to_clusters"] * 0.75,
        )

    def test_rejects_invalid_hypotheses_and_costs(self):
        for kwargs in (
            {"target_uplift": 0},
            {"target_uplift": 0.3, "expected_discordance": 0.2},
            {"expected_discordance": 1.1},
            {"alpha": 0},
            {"power": 1},
            {"mean_cluster_size": 0},
            {"intracluster_correlation": -0.1},
            {"missing_outcome_fraction": 1},
            {"grade_cost_per_task": -1},
            {"target_uplift": float("nan")},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    paired_binary_pilot_plan(**kwargs)


if __name__ == "__main__":
    unittest.main()
