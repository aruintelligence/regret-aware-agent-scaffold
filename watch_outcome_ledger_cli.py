"""Anchor a closed issued cohort, reconcile grades, and verify exact evidence."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from watch_outcome_ledger import (
    create_issued_checkpoint, reconcile_issued_grades, verify_complete_report,
)


def _key(name):
    key = os.environ.get(name, "").encode("utf-8")
    if len(key) < 32:
        raise ValueError(name + " must contain at least 32 bytes")
    return key


def _rows(path):
    source = Path(path)
    if source.stat().st_size > 10_000_000:
        raise ValueError("input JSONL exceeds 10 MB")
    with source.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _object(path):
    source = Path(path)
    if source.stat().st_size > 2_000_000:
        raise ValueError("JSON document exceeds 2 MB")
    return json.loads(source.read_text(encoding="utf-8"))


def _exclusive_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    if os.name == "posix":
        os.chmod(target, 0o600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    anchor = sub.add_parser("anchor", help="Sign the closed issued cohort with the ingress key")
    anchor.add_argument("--issued-jsonl", required=True)
    anchor.add_argument("--cohort-id", required=True)
    anchor.add_argument("--groups", required=True)
    anchor.add_argument("--model-revision", required=True)
    anchor.add_argument("--rubric-id", required=True)
    anchor.add_argument("--target-error", type=float, default=0.1)
    anchor.add_argument("--alpha", type=float, default=0.05)
    anchor.add_argument("--out", required=True)
    for name in ("reconcile", "verify"):
        cmd = sub.add_parser(name, help="Join grades to a signed issuance checkpoint")
        cmd.add_argument("--issued-jsonl", required=True)
        cmd.add_argument("--grades-jsonl", required=True)
        cmd.add_argument("--checkpoint", required=True)
        if name == "reconcile":
            cmd.add_argument("--out", required=True)
        else:
            cmd.add_argument("--report", required=True)
            cmd.add_argument("--expected-checkpoint-hmac", required=True,
                             help="Current HMAC supplied by the independently controlled ingress ledger")
    args = parser.parse_args()
    try:
        ingress_key = _key("ARU_ISSUED_SIGNING_KEY")
        if args.command == "anchor":
            checkpoint = create_issued_checkpoint(
                _rows(args.issued_jsonl), cohort_id=args.cohort_id,
                groups=[group.strip() for group in args.groups.split(",")],
                model_revision=args.model_revision, rubric_id=args.rubric_id,
                ingress_key=ingress_key, target_error=args.target_error, alpha=args.alpha,
            )
            _exclusive_json(args.out, checkpoint)
            print(json.dumps({"checkpoint": args.out,
                              "issued_count": checkpoint["issued_count"],
                              "ingress_hmac_sha256": checkpoint["ingress_hmac_sha256"]}))
            return
        watch_key = _key("ARU_WATCH_SIGNING_KEY")
        issued = _rows(args.issued_jsonl)
        grades = _rows(args.grades_jsonl)
        checkpoint = _object(args.checkpoint)
        if args.command == "reconcile":
            result = reconcile_issued_grades(
                issued, grades, checkpoint=checkpoint,
                ingress_key=ingress_key, watch_key=watch_key,
            )
            if result["status"] == "pending_grades":
                print(json.dumps(result, sort_keys=True))
                raise SystemExit(3)
            report = result["report"]
            _exclusive_json(args.out, report)
            print(json.dumps({"report": args.out, "status": "complete",
                              "issued_count": report["issued_count"],
                              "degradation_alarm": report["degradation_alarm"],
                              "automatic_rollout_permitted": False}))
            if report["degradation_alarm"]:
                raise SystemExit(10)
        else:
            report = _object(args.report)
            valid = (isinstance(checkpoint, dict)
                     and checkpoint.get("ingress_hmac_sha256") == args.expected_checkpoint_hmac
                     and verify_complete_report(
                         report, issued, grades, checkpoint=checkpoint,
                         ingress_key=ingress_key, watch_key=watch_key,
                     ))
            print(json.dumps({"exact_ledger_and_watch_verified": bool(valid)}))
            if not valid:
                raise SystemExit(1)
    except (ValueError, TypeError, OSError, json.JSONDecodeError) as error:
        parser.exit(4, "error: " + str(error) + "\n")


if __name__ == "__main__":
    main()
