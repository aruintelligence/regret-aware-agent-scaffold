"""Local-only ĀRU source grant/revocation CLI.

Operator must obtain and verify actual consent through a separate process.
Source text and owner identifiers are read from local files, not command
arguments. Never store the registry key inside Git or the SQLite database.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sovereign_source_registry import (
    authorize_source, grant_source, revoke_source, verify_registry,
)


def _read(path):
    return Path(path).read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    grant = sub.add_parser("grant", help="Attest time-limited purpose-scoped source consent")
    grant.add_argument("--db", required=True)
    grant.add_argument("--source-file", required=True)
    grant.add_argument("--owner-file", required=True)
    grant.add_argument("--purpose", default="qa_source")
    grant.add_argument("--ttl-seconds", type=int, default=3600)
    grant.add_argument("--consent-asserted", action="store_true")
    revoke = sub.add_parser("revoke", help="Revoke a source grant")
    revoke.add_argument("--db", required=True)
    revoke.add_argument("--grant-id", required=True)
    revoke.add_argument("--owner-file", required=True)
    check = sub.add_parser("check", help="Check source authorization; nonzero if denied")
    check.add_argument("--db", required=True)
    check.add_argument("--source-file", required=True)
    check.add_argument("--owner-file", required=True)
    check.add_argument("--purpose", default="qa_source")
    verify = sub.add_parser("verify", help="Verify complete local HMAC consent event chain")
    verify.add_argument("--db", required=True)
    args = parser.parse_args()
    key = os.environ.get("ARU_SOURCE_REGISTRY_KEY", "").encode("utf-8")
    if len(key) < 32:
        parser.error("ARU_SOURCE_REGISTRY_KEY must be set to a separate 32+ byte secret")
    if args.command == "grant":
        result = grant_source(
            args.db, _read(args.source_file),
            owner=_read(args.owner_file).strip(),
            purpose=args.purpose, ttl_seconds=args.ttl_seconds,
            key=key, consent_asserted=args.consent_asserted,
        )
    elif args.command == "revoke":
        result = revoke_source(
            args.db, args.grant_id,
            owner=_read(args.owner_file).strip(), key=key,
        )
    elif args.command == "check":
        result = authorize_source(
            args.db, _read(args.source_file),
            owner=_read(args.owner_file).strip(),
            purpose=args.purpose, key=key,
        )
    else:
        result = verify_registry(args.db, key=key)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "check" and not result["authorized"]:
        raise SystemExit(1)
    if args.command == "verify" and not result["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
