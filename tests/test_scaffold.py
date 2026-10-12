import tempfile
import unittest
from pathlib import Path

from regret_aware_scaffold import (
    MasterAgentRuntime,
    PlannerRecord,
    RouteMemory,
    TaskType,
    predict_route_quality,
    score_run_quality,
    task_signature_from_state,
)


PROMPT = "Design and research a Python master agent scaffold with route memory and evaluation."


class ScaffoldRegressionTests(unittest.TestCase):
    def test_route_memory_is_retrieved_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "memory.json")
            first = MasterAgentRuntime(memory_path=path).run(PROMPT)
            signature = task_signature_from_state(first)
            stored = RouteMemory.load_json(path).route_history[signature]
            self.assertTrue(stored)
            self.assertTrue(all("planner_node" not in record.path for record in stored))
            self.assertTrue(all(record.path for record in stored))

            second = MasterAgentRuntime(memory_path=path).run(PROMPT)
            self.assertIn(first.route.planned_path, second.memory_hints["best_known_paths"])
            self.assertLessEqual(len(second.route.selected_path), len(first.route.selected_path))

    def test_quality_penalizes_current_route_and_retries_once(self):
        runtime = MasterAgentRuntime()
        state = runtime.new_state(PROMPT)
        state.task_type = TaskType.DESIGN
        state.route.planned_path = ["systems_design_node", "coherence_node", "compression_node"]
        state.route.selected_path = ["planner_node"] * 48
        state.route.loop_count = 2
        runtime.nodes["evaluator_node"].process(state)
        expected = score_run_quality(state.metrics, path_length=3, loop_count=2)
        self.assertAlmostEqual(state.evaluation.overall_quality, expected)

    def test_calibration_moves_both_directions_at_matching_depth(self):
        runtime = MasterAgentRuntime()
        state = runtime.new_state(PROMPT)
        state.task_type = TaskType.DESIGN
        state.context.domain_tags = ["python"]
        route = ["systems_design_node", "coherence_node", "compression_node"]
        signature = task_signature_from_state(state)
        baseline = predict_route_quality(state, route, runtime.memory)

        def record(actual, depth):
            return PlannerRecord(
                task_signature=signature, selected_route=route,
                predicted_quality=baseline, predicted_cost=0.09,
                predicted_score=baseline - 0.09, actual_quality=actual,
                prediction_error=actual - (baseline - 0.09), loop_count=depth,
            )

        runtime.memory.planner_history[signature] = [record(0.95, 1)]
        self.assertAlmostEqual(predict_route_quality(state, route, runtime.memory), baseline)
        runtime.memory.planner_history[signature] = [record(0.95, 0)]
        self.assertGreater(predict_route_quality(state, route, runtime.memory), baseline)
        runtime.memory.planner_history[signature] = [record(0.05, 0)]
        self.assertLess(predict_route_quality(state, route, runtime.memory), baseline)

    def test_reported_confidence_never_exceeds_internal_quality(self):
        state = MasterAgentRuntime().run(PROMPT)
        self.assertLessEqual(state.final_output.confidence, state.evaluation.overall_quality)
        self.assertIn("does not execute external tools", state.final_output.answer)


if __name__ == "__main__":
    unittest.main()
