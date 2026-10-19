import unittest

from extractive_qa_engine import best_context_span


class ExtractiveSpanTests(unittest.TestCase):
    def test_selects_best_context_span_not_question_or_special_tokens(self):
        context = "The telescope was in Oregon."
        offsets = [(0, 0), (0, 3), (0, 0), (0, 3), (4, 13),
                   (14, 17), (18, 20), (21, 27), (27, 28)]
        roles = [None, 0, None, 1, 1, 1, 1, 1, 1]
        starts = [100, 200, 100, 0, 0, 0, 0, 8, 0]
        ends = [100, 200, 100, 0, 0, 0, 0, 8, 0]
        result = best_context_span(context, offsets, roles, starts, ends)
        self.assertEqual(context[result["start"]:result["end"]], "Oregon")
        self.assertEqual(result["start"], 21)
        self.assertEqual(result["end"], 27)

    def test_rejects_noncontext_and_invalid_offsets(self):
        self.assertIsNone(best_context_span(
            "Oregon", [(0, 6), (0, 0)], [0, None], [1, 2], [1, 2],
        ))
        self.assertIsNone(best_context_span(
            "Oregon", [(-1, 6), (0, 20)], [1, 1], [1, 2], [1, 2],
        ))

    def test_finite_logits_and_consistent_token_lengths_required(self):
        with self.assertRaises(ValueError):
            best_context_span("abc", [(0, 3)], [1, 1], [1], [1])
        self.assertIsNone(best_context_span(
            "abc", [(0, 3)], [1], [float("nan")], [1],
        ))

    def test_shortest_span_wins_on_tied_logits(self):
        context = "The Oregon coast"
        result = best_context_span(
            context,
            [(0, 3), (4, 10), (11, 16)],
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        )
        self.assertEqual(context[result["start"]:result["end"]], "Oregon")

    def test_refuses_overlong_spans(self):
        context = "a" * 210
        result = best_context_span(
            context,
            [(0, 105), (105, 210)],
            [1, 1],
            [10, 0],
            [0, 10],
            max_answer_tokens=2,
        )
        self.assertIsNotNone(result)
        self.assertLessEqual(result["end"] - result["start"], 200)

    def test_invalid_window_size_rejected(self):
        with self.assertRaises(ValueError):
            best_context_span("abc", [(0, 3)], [1], [1], [1], max_answer_tokens=0)


if __name__ == "__main__":
    unittest.main()
