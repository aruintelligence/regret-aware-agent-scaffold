import unittest

from paired_inference import exact_mcnemar


class PairedInferenceTests(unittest.TestCase):
    def test_perfectly_balanced_discordant_pairs(self):
        result = exact_mcnemar(5, 5)
        self.assertEqual(result["exact_two_sided_p"], 1.0)
        self.assertEqual(result["discordant_pairs"], 10)

    def test_all_wins_one_side_matches_closed_form(self):
        # 10 discordant pairs all favor candidate: p=2/2^10
        result = exact_mcnemar(0, 10)
        self.assertAlmostEqual(result["exact_two_sided_p"], 2 / 1024)
        self.assertEqual(result["candidate_win_fraction_among_discordant"], 1)

    def test_zero_discordance_is_non_significant(self):
        result = exact_mcnemar(0, 0)
        self.assertEqual(result["exact_two_sided_p"], 1)
        self.assertIsNone(result["candidate_win_fraction_among_discordant"])

    def test_symmetric_under_swapping_algorithms(self):
        self.assertEqual(
            exact_mcnemar(17, 9)["exact_two_sided_p"],
            exact_mcnemar(9, 17)["exact_two_sided_p"],
        )

    def test_large_counts_stable_without_float_overflow(self):
        result = exact_mcnemar(400, 650)
        self.assertTrue(0 <= result["exact_two_sided_p"] <= 1)
        self.assertEqual(result["discordant_pairs"], 1050)

    def test_invalid_counts_rejected(self):
        for values in ((-1, 0), (0.1, 4), (True, 1)):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    exact_mcnemar(*values)


if __name__ == "__main__":
    unittest.main()
