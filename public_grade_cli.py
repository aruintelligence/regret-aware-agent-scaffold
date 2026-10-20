"""Sign grades privately and reconcile them with only a pinned public key."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from grade_attestation_ledger import _plain_grades
from public_grade_ledger import (
    audit_public_grades, close_public_grades, public_checkpoint_fingerprint,
    public_key_fingerprint, sign_public_grade, verify_public_grade_log,
)
from watch_outcome_ledger import reconcile_issued_grades


def _read_object(path, limit=2_000_000):
    path = Path(path)
    if path.stat().st_size > limit:
        raise ValueError("input JSON exceeds size limit")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path):
    path = Path(path)
    if path.stat().st_size > 10_000_000:
        raise ValueError("input JSONL exceeds 10 MB")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_private(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL refuses overwrites and O_NOFOLLOW refuses a final symlink on POSIX.
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(target, flags, 0o600), "w", encoding="utf-8") as handle:
        json.dump(data, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _raw_key(path, name):
    document = _read_object(path, limit=1024)
    field = name + "_key_hex"
    if (not isinstance(document, dict)
            or set(document) != {"kind", field}
            or document["kind"] != "aru_ed25519_" + name + "_key"
            or not isinstance(document[field], str)
            or len(document[field]) != 64
            or any(c not in "0123456789abcdef" for c in document[field])):
        raise ValueError("invalid " + name + " key file")
    return bytes.fromhex(document[field])


def _hmac_key(name):
    key = os.environ.get(name, "").encode("utf-8")
    if len(key) < 32:
        raise ValueError(name + " must contain at least 32 bytes")
    return key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    keygen = sub.add_parser("keygen", help="Create a grader-only secret and distributable public key")
    keygen.add_argument("--private-out", required=True)
    keygen.add_argument("--public-out", required=True)
    grade = sub.add_parser("grade", help="Sign a single binary grade")
    grade.add_argument("--private-key", required=True)
    grade.add_argument("--grade-json", required=True)
    grade.add_argument("--grade-sequence", type=int, required=True)
    grade.add_argument("--cohort-id", required=True)
    grade.add_argument("--model-revision", required=True)
    grade.add_argument("--out", required=True)
    close = sub.add_parser("close", help="Sign an independently retained closed grader log")
    close.add_argument("--private-key", required=True)
    close.add_argument("--attestations-jsonl", required=True)
    close.add_argument("--cohort-id", required=True)
    close.add_argument("--model-revision", required=True)
    close.add_argument("--rubric-id", required=True)
    close.add_argument("--out", required=True)
    for command in ("verify", "reconcile"):
        verify = sub.add_parser(
            command, help="Verify a grader log without secrets" if command == "verify"
            else "Verify grades and sign a watch without grader private key")
        verify.add_argument("--public-key", required=True)
        verify.add_argument("--expected-public-key-sha256", required=True)
        verify.add_argument("--expected-current-grade-sha256", required=True)
        verify.add_argument("--attestations-jsonl", required=True)
        verify.add_argument("--grade-checkpoint", required=True)
        if command == "reconcile":
            verify.add_argument("--expected-current-ingress-hmac", required=True)
            verify.add_argument("--issued-jsonl", required=True)
            verify.add_argument("--ingress-checkpoint", required=True)
            verify.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        if args.command == "keygen":
            if Path(args.private_out).resolve() == Path(args.public_out).resolve():
                raise ValueError("private and public paths must be different")
            if Path(args.private_out).exists() or Path(args.public_out).exists():
                raise FileExistsError("key output path already exists")
            private = Ed25519PrivateKey.generate()
            private_bytes = private.private_bytes_raw()
            public_bytes = private.public_key().public_bytes_raw()
            # Neither secret nor raw key material is printed to stdout.
            _write_private(args.private_out, {"kind": "aru_ed25519_private_key",
                                              "private_key_hex": private_bytes.hex()})
            _write_private(args.public_out, {"kind": "aru_ed25519_public_key",
                                             "public_key_hex": public_bytes.hex()})
            print(json.dumps({"private_key": args.private_out, "public_key": args.public_out,
                              "public_key_sha256": public_key_fingerprint(public_bytes)}))
            return
        if args.command == "grade":
            signed = sign_public_grade(
                _read_object(args.grade_json), grade_sequence=args.grade_sequence,
                cohort_id=args.cohort_id, model_revision=args.model_revision,
                grader_private_key=_raw_key(args.private_key, "private"),
            )
            _write_private(args.out, signed)
            print(json.dumps({"attestation": args.out, "grade_sequence": signed["grade_sequence"]}))
            return
        if args.command == "close":
            checkpoint = close_public_grades(
                _read_rows(args.attestations_jsonl), cohort_id=args.cohort_id,
                model_revision=args.model_revision, rubric_id=args.rubric_id,
                grader_private_key=_raw_key(args.private_key, "private"),
            )
            _write_private(args.out, checkpoint)
            print(json.dumps({"checkpoint": args.out, "grade_count": checkpoint["grade_count"],
                              "grade_checkpoint_sha256": public_checkpoint_fingerprint(checkpoint)}))
            return
        public_key = _raw_key(args.public_key, "public")
        attestations = _read_rows(args.attestations_jsonl)
        grade_checkpoint = _read_object(args.grade_checkpoint)
        if args.command == "verify":
            result = verify_public_grade_log(
                attestations, grade_checkpoint=grade_checkpoint,
                grader_public_key=public_key,
                expected_grader_public_key_sha256=args.expected_public_key_sha256,
                expected_current_grade_checkpoint_sha256=args.expected_current_grade_sha256,
            )
            print(json.dumps(result, sort_keys=True))
            return
        issued = _read_rows(args.issued_jsonl)
        ingress_checkpoint = _read_object(args.ingress_checkpoint)
        ingress_key = _hmac_key("ARU_ISSUED_SIGNING_KEY")
        watch_key = _hmac_key("ARU_WATCH_SIGNING_KEY")
        audit = audit_public_grades(
            attestations, issued, grade_checkpoint=grade_checkpoint,
            ingress_checkpoint=ingress_checkpoint, grader_public_key=public_key,
            expected_grader_public_key_sha256=args.expected_public_key_sha256,
            expected_current_grade_checkpoint_sha256=args.expected_current_grade_sha256,
            ingress_key=ingress_key,
            expected_current_ingress_hmac_sha256=args.expected_current_ingress_hmac,
        )
        if audit["status"] != "complete":
            print(json.dumps(audit, sort_keys=True))
            raise SystemExit(3)
        result = reconcile_issued_grades(
            issued, _plain_grades(attestations), checkpoint=ingress_checkpoint,
            ingress_key=ingress_key, watch_key=watch_key,
        )
        if result["status"] != "complete":
            print(json.dumps(result, sort_keys=True))
            raise SystemExit(3)
        report = result["report"]
        _write_private(args.out, report)
        print(json.dumps({"report": args.out, "status": "complete",
                          "graded_count": report["graded_count"],
                          "degradation_alarm": report["degradation_alarm"],
                          "automatic_rollout_permitted": False}))
        if report["degradation_alarm"]:
            raise SystemExit(10)
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as error:
        parser.exit(4, "error: " + str(error) + "\n")


if __name__ == "__main__":
    main()
