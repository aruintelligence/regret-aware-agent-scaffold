"""CLI for signed prospective risk deterioration monitoring.

Only aggregate statistics and an ordered-outcome SHA-256 digest are emitted.
The source JSONL remains local. No hidden background monitoring is performed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from anytime_risk_watch import sign_watch_report, verify_watch_report, watch_risk_stream


def _key():
    key = os.environ.get("ARU_WATCH_SIGNING_KEY", "").encode("utf-8")
    if len(key) < 32:
        raise ValueError("ARU_WATCH_SIGNING_KEY must contain at least 32 bytes")
    return key


def _read_rows(path):
    source = Path(path)
    if source.stat().st_size > 10_000_000:
        raise ValueError("outcome JSONL exceeds 10 MB")
    with source.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_policy(path):
    source = Path(path)
    if source.stat().st_size > 2_000_000:
        raise ValueError("watch report exceeds 2 MB")
    return json.loads(source.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("observe", help="Sign an aggregate prospective monitoring report")
    fit.add_argument("--outcomes-jsonl", required=True)
    fit.add_argument("--groups", required=True)
    fit.add_argument("--model-revision", required=True)
    fit.add_argument("--rubric-id", required=True)
    fit.add_argument("--target-error", type=float, default=0.1)
    fit.add_argument("--alpha", type=float, default=0.05)
    fit.add_argument("--out", required=True)
    check = sub.add_parser("verify", help="Verify signed watch report")
    check.add_argument("--report", required=True)
    args = parser.parse_args()
    try:
        key = _key()
        if args.command == "observe":
            report = watch_risk_stream(
                _read_rows(args.outcomes_jsonl),
                groups=[x.strip() for x in args.groups.split(",")],
                model_revision=args.model_revision, rubric_id=args.rubric_id,
                target_error=args.target_error, alpha=args.alpha,
            )
            signed = sign_watch_report(report, key=key)
            target = Path(args.out)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as handle:
                json.dump(signed, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
            if os.name == "posix":
                os.chmod(target, 0o600)
            print(json.dumps({
                "report": str(target), "outcomes": report["outcome_count"],
                "degradation_alarm": report["degradation_alarm"],
                "alarm_groups": report["alarm_groups"],
                "automatic_rollout_permitted": False,
            }))
            if report["degradation_alarm"]:
                raise SystemExit(2)
        else:
            valid = verify_watch_report(_read_policy(args.report), key=key)
            print(json.dumps({"watch_hmac_verified": valid}))
            if not valid:
                raise SystemExit(1)
    except (ValueError, TypeError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
