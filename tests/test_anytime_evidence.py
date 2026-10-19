import math
import unittest

from anytime_evidence import paired_e_process


class AnytimeEvidenceTests(unittest.TestCase):
    def test_consistent_candidate_wins_cross_predeclared_threshold(self):
        result = paired_e_process([(0, 1)] * 80, alpha=0.05)
        self.assertTrue(result["ever_crossed_threshold"])
        self.assertIsNotNone(result["first_crossing_pair"])
        self.assertLess(result["first_crossing_pair"], 80)
        self.assertFalse(result["automatic_rollout_permitted"])

    def test_consistent_losses_do_not_cross(self):
        result = paired_e_process([(1, 0)] * 80)
        self.assertFalse(result["ever_crossed_threshold"])
        self.assertLess(result["final_e_value"], 1)

    def test_ties_do_not_change_wealth(self):
        result = paired_e_process([(1, 1), (0, 0)] * 30)
        self.assertAlmostEqual(result["final_e_value"], 1)
        self.assertAlmostEqual(result["observed_paired_difference"], 0)

    def test_single_bet_has_exact_product(self):
        result = paired_e_process([(0, 1)] * 3 + [(1, 0)], bets=(0.5,))
        self.assertAlmostEqual(result["final_e_value"], (1.5 ** 3) * 0.5)

    def test_final_value_is_order_invariant_but_first_crossing_may_not_be(self):
        a = paired_e_process([(0, 1)] * 30 + [(1, 0)] * 10)
        b = paired_e_process([(1, 0)] * 10 + [(0, 1)] * 30)
        self.assertAlmostEqual(a["final_log_e_value"], b["final_log_e_value"])
        self.assertNotEqual(a["first_crossing_pair"], b["first_crossing_pair"])

    def test_invalid_data_and_bets_fail_closed(self):
        for rows in ([], [(1, 2)], [(True, 0)], [(1,)], ["bad"]):
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    paired_e_process(rows)
        with self.assertRaises(ValueError):
            paired_e_process([(0, 1)], bets=(1.1,))
        with self.assertRaises(ValueError):
            paired_e_process([(0, 1)], alpha=0)

    def test_long_stream_numerically_stable(self):
        result = paired_e_process([(0, 1)] * 2000)
        self.assertTrue(math.isfinite(result["final_log_e_value"]))
        self.assertTrue(result["ever_crossed_threshold"])


if __name__ == "__main__":
    unittest.main()
