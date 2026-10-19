"""Prove plan precedence, real-call accounting and fail-closed verification."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from grounding_contract import ground_answer
from prospective_ledger_audit import (
    prepare_cohort, run_cohort, verify_cohort,
)
from watch_outcome_ledger import create_issued_checkpoint

INGRESS_KEY = b"prospective-fixture-ingress-key-32-bytes"
WATCH_KEY = b"prospective-fixture-monitor-key-32-bytes"


class FakeQA:
    model_id = "test-real-call-adapter"
    revision = "frozen-fake-revision"

    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def answer(self, context, question):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("record this failed call as an error")
        proof = ground_answer("Oregon", context, model_revision=self.revision)
        return {"accepted": proof["accepted"], "answer": proof["answer"],
                "proof": proof, "model_revision": self.revision}


def fixture(path):
    short = "The telescope was in Oregon."
    long = short + " It was used for astronomy." * 30
    data = {"version": "1.1", "data": [{"title": "test", "paragraphs": [
        {"context": context, "qas": [{"id": f"test-{i:02d}",
          "question": "Where was the telescope?",
          "answers": [{"text": "Oregon", "answer_start": context.index("Oregon")}]}]}
        for i, context in enumerate([short, long] * 8)
    ]}]}
    raw = json.dumps(data).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


class ProspectiveLedgerAuditTests(unittest.TestCase):
    def _setup(self, folder):
        root = Path(folder)
        dataset = root / "dataset.json"
        digest = fixture(dataset)
        prepared = root / "prepared"
        checkpoint = prepare_cohort(
            dataset, prepared, ingress_key=INGRESS_KEY,
            expected_dataset_sha256=digest, offset=2, limit=12,
            model_revision=FakeQA.revision,
        )
        return dataset, digest, prepared, checkpoint

    def _run(self, dataset, digest, prepared, checkpoint, target, qa):
        return run_cohort(
            dataset, prepared, target, ingress_key=INGRESS_KEY,
            watch_key=WATCH_KEY, expected_dataset_sha256=digest,
            expected_checkpoint_hmac=checkpoint["ingress_hmac_sha256"],
            offset=2, limit=12, model_revision=FakeQA.revision,
            qa_model=FakeQA.model_id, engine=qa,
            acknowledge_model_calls=True,
        )

    def test_prepared_before_inference_and_exact_complete_recheck(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset, digest, prepared, checkpoint = self._setup(folder)
            self.assertEqual(checkpoint["issued_count"], 12)
            self.assertFalse((prepared / "grades.jsonl").exists())
            qa = FakeQA(fail_at=4)
            target = Path(folder) / "run"
            report = self._run(dataset, digest, prepared, checkpoint, target, qa)
            self.assertEqual(qa.calls, 12)
            self.assertEqual(report["requested"], report["graded"])
            self.assertEqual(report["errors"], 1)
            self.assertEqual(report["model_call_failures_counted_as_errors"], 1)
            self.assertFalse(report["independent_ingress_operator"])
            self.assertFalse(report["automatic_rollout_permitted"])
            checked = verify_cohort(
                dataset, prepared, target, ingress_key=INGRESS_KEY,
                watch_key=WATCH_KEY, expected_dataset_sha256=digest,
                expected_checkpoint_hmac=checkpoint["ingress_hmac_sha256"],
                offset=2, limit=12, model_revision=FakeQA.revision,
                qa_model=FakeQA.model_id,
            )
            self.assertEqual(checked, report)

    def test_invalid_or_changed_plan_blocks_all_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset, digest, prepared, checkpoint = self._setup(folder)
            for change in ("stale", "manifest", "checkpoint", "dataset"):
                with self.subTest(change=change):
                    qa = FakeQA()
                    source = prepared / "issued.jsonl"
                    anchor = prepared / "checkpoint.json"
                    original_source = source.read_text()
                    original_anchor = anchor.read_text()
                    provided_digest = digest
                    expected_hmac = checkpoint["ingress_hmac_sha256"]
                    if change == "stale":
                        expected_hmac = "0" * 64
                    elif change == "manifest":
                        source.write_text(original_source.replace("short_source", "long_source", 1))
                    elif change == "checkpoint":
                        anchor.write_text(original_anchor.replace("\"issued_count\": 12", "\"issued_count\": 11"))
                    else:
                        provided_digest = "0" * 64
                    try:
                        with self.assertRaises(ValueError):
                            run_cohort(
                                dataset, prepared, Path(folder) / f"run-{change}",
                                ingress_key=INGRESS_KEY, watch_key=WATCH_KEY,
                                expected_checkpoint_hmac=expected_hmac,
                                expected_dataset_sha256=provided_digest,
                                offset=2, limit=12, model_revision=FakeQA.revision,
                                qa_model=FakeQA.model_id, engine=qa,
                                acknowledge_model_calls=True,
                            )
                        self.assertEqual(qa.calls, 0)
                    finally:
                        source.write_text(original_source)
                        anchor.write_text(original_anchor)

    def test_missing_or_rewritten_grade_fails_exact_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset, digest, prepared, checkpoint = self._setup(folder)
            target = Path(folder) / "run"
            self._run(dataset, digest, prepared, checkpoint, target, FakeQA())
            grades_path = target / "grades.jsonl"
            original = grades_path.read_text()
            rows = [json.loads(line) for line in original.splitlines()]
            for variant in (rows[:-1], [{**rows[0], "error": True}] + rows[1:]):
                with self.subTest(variant=len(variant)):
                    grades_path.write_text("".join(json.dumps(row) + "\n" for row in variant))
                    with self.assertRaises(ValueError):
                        verify_cohort(
                            dataset, prepared, target, ingress_key=INGRESS_KEY,
                            watch_key=WATCH_KEY, expected_dataset_sha256=digest,
                            expected_checkpoint_hmac=checkpoint["ingress_hmac_sha256"],
                            offset=2, limit=12, model_revision=FakeQA.revision,
                            qa_model=FakeQA.model_id,
                        )
            grades_path.write_text(original)
            self.assertTrue(verify_cohort(
                dataset, prepared, target, ingress_key=INGRESS_KEY,
                watch_key=WATCH_KEY, expected_dataset_sha256=digest,
                expected_checkpoint_hmac=checkpoint["ingress_hmac_sha256"],
                offset=2, limit=12, model_revision=FakeQA.revision,
                qa_model=FakeQA.model_id,
            ))

    def test_resigned_looser_watch_target_blocks_before_model_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset, digest, prepared, _ = self._setup(folder)
            issued = [json.loads(line) for line in (prepared / "issued.jsonl").read_text().splitlines()]
            looser = create_issued_checkpoint(
                issued, cohort_id="squad-680-799-prospective-ledger-v1",
                groups=["short_source", "long_source"],
                model_revision=FakeQA.revision,
                rubric_id="squad_v1.1_human_gold_normalized_f1_at_least_0.5_v1",
                ingress_key=INGRESS_KEY, target_error=0.9,
            )
            (prepared / "checkpoint.json").write_text(json.dumps(looser))
            plan_path = prepared / "plan.json"
            plan = json.loads(plan_path.read_text())
            plan["ingress_checkpoint_hmac_sha256"] = looser["ingress_hmac_sha256"]
            plan_path.write_text(json.dumps(plan))
            qa = FakeQA()
            with self.assertRaises(ValueError):
                self._run(dataset, digest, prepared, looser, Path(folder) / "run", qa)
            self.assertEqual(qa.calls, 0)


if __name__ == "__main__":
    unittest.main()
