"""Operator CLI for ĀRU's research-only selective answer risk gate.

Never writes raw calibration rows into reports. The signing secret is read
only from ARU_RISK_SIGNING_KEY, never a command-line argument.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from selective_risk_gate import (
    apply_selective_risk_gate, fit_selective_risk_policy,
    sign_risk_policy, verify_risk_policy,
)


def _key():
    key = os.environ.get("ARU_RISK_SIGNING_KEY", "").encode("utf-8")
    if len(key) < 32:
        raise ValueError("ARU_RISK_SIGNING_KEY must contain at least 32 bytes")
    return key


def _read_json(path, *, max_bytes=10_000_000):
    file = Path(path)
    if file.stat().st_size > max_bytes:
        raise ValueError("input too large")
    return json.loads(file.read_text(encoding="utf-8"))


def _load_jsonl(path):
    file = Path(path)
    if file.stat().st_size > 10_000_000:
        raise ValueError("calibration file too large")
    rows = []
    with file.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_new(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    if os.name == "posix":
        os.chmod(target, 0o600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="Fit and HMAC-sign a risk screen using independent calibration labels")
    fit.add_argument("--calibration-jsonl", required=True)
    fit.add_argument("--thresholds", required=True, help="Predeclared comma-separated increasing scores")
    fit.add_argument("--groups", required=True, help="Predeclared comma-separated domain identifiers")
    fit.add_argument("--target-error", type=float, default=0.1)
    fit.add_argument("--alpha", type=float, default=0.05)
    fit.add_argument("--min-group-support", type=int, default=30)
    fit.add_argument("--model-revision", required=True)
    fit.add_argument("--rubric-id", required=True)
    fit.add_argument("--calibration-set-id", required=True)
    fit.add_argument("--out", required=True)
    check = sub.add_parser("check", help="Verify signed risk policy and independent source certificate")
    check.add_argument("--policy", required=True)
    check.add_argument("--score", type=float, required=True)
    check.add_argument("--group", required=True)
    check.add_argument("--source-file", required=True)
    check.add_argument("--proof-file", required=True)
    check.add_argument("--model-revision", required=True)
    check.add_argument("--rubric-id", required=True)
    verify = sub.add_parser("verify", help="Verify policy HMAC without accessing calibration data")
    verify.add_argument("--policy", required=True)
    args = parser.parse_args()
    try:
        key = _key()
        if args.command == "fit":
            thresholds = [float(x) for x in args.thresholds.split(",")]
            groups = [x.strip() for x in args.groups.split(",")]
            policy = fit_selective_risk_policy(
                _load_jsonl(args.calibration_jsonl),
                thresholds=thresholds, groups=groups,
                target_error=args.target_error, alpha=args.alpha,
                min_group_support=args.min_group_support,
                model_revision=args.model_revision, rubric_id=args.rubric_id,
                calibration_set_id=args.calibration_set_id,
            )
            signed = sign_risk_policy(policy, key=key)
            _write_new(args.out, signed)
            print(json.dumps({
                "policy_file": args.out,
                "calibration_rows": policy["calibration_rows"],
                "conditional_risk_screen_passed": policy["conditional_risk_screen_passed"],
                "selected_threshold": policy["selected_threshold"],
                "selected_coverage": policy["selected_coverage"],
                "automatic_rollout_permitted": False,
            }, sort_keys=True))
        elif args.command == "verify":
            valid = verify_risk_policy(_read_json(args.policy), key=key)
            print(json.dumps({"policy_hmac_verified": valid}))
            if not valid:
                raise SystemExit(1)
        else:
            result = apply_selective_risk_gate(
                _read_json(args.policy), key=key,
                score=args.score, group=args.group,
                source_proof=_read_json(args.proof_file),
                source_text=Path(args.source_file).read_text(encoding="utf-8"),
                model_revision=args.model_revision, rubric_id=args.rubric_id,
            )
            print(json.dumps(result, sort_keys=True, allow_nan=False))
            if not result["accepted"]:
                raise SystemExit(1)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
