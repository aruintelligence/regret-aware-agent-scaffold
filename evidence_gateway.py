"""Loopback-only research HTTP API for consented ĀRU evidence ingestion.

POST /v1/decisions   -> validated, idempotent decision event
POST /v1/labels      -> validated, independently graded outcome
GET  /v1/chain       -> HMAC audit verification
GET  /health         -> liveness (no stored data exposed)

This is NOT an internet-facing production service. There is no TLS, SSO,
RBAC, tenant isolation, encryption at rest, or consent verification.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import sqlite3
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from evidence_gateway_store import (
    EvidenceConflict, export_jsonl, record_event, verify_chain,
)
from evidence_lab import evaluate_experiment
from evidence_dashboard import save_dashboard
from evidence_audit import build_manifest, save_manifest
from evidence_checkpoint import create_checkpoint, verify_checkpoint


def _load_secret(name, min_bytes):
    value = os.environ.get(name, "")
    if len(value.encode("utf-8")) < min_bytes:
        raise ValueError(name + " environment variable must have at least " + str(min_bytes) + " bytes")
    return value.encode("utf-8")


def make_handler(db_path, api_key, hmac_key):
    if not isinstance(api_key, bytes) or len(api_key) < 24:
        raise ValueError("API key must be at least 24 bytes")
    if not isinstance(hmac_key, bytes) or len(hmac_key) < 16:
        raise ValueError("HMAC key must be at least 16 bytes")

    class Handler(BaseHTTPRequestHandler):
        server_version = "ARU-Evidence-Gateway/0.1"
        sys_version = ""

        def log_message(self, format, *args):
            # Do not log request URLs, tokens, or sensitive payloads.
            pass

        def _respond(self, status, payload):
            raw = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(raw)

        def _authorized(self):
            header = self.headers.get("Authorization", "")
            supplied = header[7:].encode("utf-8") if header.startswith("Bearer ") else b""
            if not hmac.compare_digest(supplied, api_key):
                self._respond(401, {"error": "unauthorized"})
                return False
            return True

        def do_GET(self):
            if self.path == "/health":
                self._respond(200, {"status": "ok", "scope": "loopback_research_only"})
                return
            if self.path != "/v1/chain":
                self._respond(404, {"error": "not_found"})
                return
            if not self._authorized():
                return
            try:
                self._respond(200, verify_chain(db_path, hmac_key=hmac_key))
            except (sqlite3.Error, OSError):
                self._respond(503, {"error": "evidence_store_unavailable"})

        def do_POST(self):
            kind = {"/v1/decisions": "decision", "/v1/labels": "label"}.get(self.path)
            if kind is None:
                self._respond(404, {"error": "not_found"})
                return
            if not self._authorized():
                return
            if self.headers.get("X-ARU-Consent") != "research-opt-in":
                self._respond(403, {"error": "explicit_research_consent_header_required"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._respond(400, {"error": "invalid_content_length"})
                return
            if length < 2 or length > 100000:
                self._respond(413, {"error": "request_size_outside_limits"})
                return
            try:
                body = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._respond(400, {"error": "invalid_json"})
                return
            try:
                result = record_event(db_path, kind, body, hmac_key=hmac_key)
            except EvidenceConflict:
                self._respond(409, {"error": "conflicting_event_id"})
                return
            except (ValueError, KeyError, TypeError):
                self._respond(422, {"error": "invalid_event_schema_or_unknown_decision"})
                return
            except sqlite3.Error:
                self._respond(503, {"error": "evidence_store_unavailable"})
                return
            self._respond(200 if result["duplicate"] else 201, result)

    return Handler


def make_server(db_path, api_key, hmac_key, *, host="127.0.0.1", port=8765):
    if host != "127.0.0.1":
        raise ValueError("research gateway currently supports IPv4 loopback only")
    return ThreadingHTTPServer((host, port), make_handler(db_path, api_key, hmac_key))


def build_report_from_gateway(db_path, out_dir, *, hmac_key,
                              baseline_policy="greedy",
                              candidate_policy="shortest",
                              replicates=300):
    """Export, evaluate, visualize and attest consented research events."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    decisions = str(target / "decisions.jsonl")
    labels = str(target / "labels.jsonl")
    report_json = target / "report.json"
    report_html = target / "index.html"
    manifest_path = target / "attestation.json"
    if any(p.exists() for p in (report_json, report_html, manifest_path)):
        raise FileExistsError("report files already exist; use fresh output directory")
    exported = export_jsonl(db_path, decisions, labels, hmac_key=hmac_key)
    if not exported["labels"]:
        raise ValueError("no external outcome labels; no report generated")
    report = evaluate_experiment(
        decisions, labels, baseline_policy=baseline_policy,
        candidate_policy=candidate_policy, replicates=replicates,
    )
    report["source"] = {
        "kind": "consented_local_gateway",
        "audit_verified_before_export": True,
        "note": "Local HMAC audit does not authenticate real-world truth of outcomes.",
    }
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    save_dashboard(report, str(report_html))
    manifest = build_manifest(decisions, labels, report_path=str(report_json),
                              secret=hmac_key)
    save_manifest(manifest, str(manifest_path))
    return {
        "report_json": str(report_json),
        "report_html": str(report_html),
        "attestation": str(manifest_path),
        "n_labeled": report["n_labeled"],
        "automatic_rollout_permitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--db", required=True)
    serve.add_argument("--port", type=int, default=8765)
    verify = sub.add_parser("verify")
    verify.add_argument("--db", required=True)
    export = sub.add_parser("export")
    export.add_argument("--db", required=True)
    export.add_argument("--decisions", required=True)
    export.add_argument("--labels", required=True)
    report = sub.add_parser("report")
    report.add_argument("--db", required=True)
    report.add_argument("--out-dir", required=True)
    report.add_argument("--baseline-policy", default="greedy")
    report.add_argument("--candidate-policy", default="shortest")
    report.add_argument("--replicates", type=int, default=300)
    anchor = sub.add_parser("anchor", help="Write independently signed audit checkpoint")
    anchor.add_argument("--db", required=True)
    anchor.add_argument("--checkpoint", required=True)
    check = sub.add_parser("verify-anchor", help="Verify historical chain against external checkpoint")
    check.add_argument("--db", required=True)
    check.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    hmac_key = _load_secret("ARU_GATEWAY_HMAC_KEY", 16)
    if args.command == "serve":
        api_key = _load_secret("ARU_GATEWAY_API_KEY", 24)
        server = make_server(args.db, api_key, hmac_key, port=args.port)
        print("ĀRU research gateway listening on loopback port " + str(server.server_port))
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    elif args.command == "verify":
        print(json.dumps(verify_chain(args.db, hmac_key=hmac_key), indent=2))
    elif args.command == "anchor":
        independent_key = _load_secret("ARU_ANCHOR_HMAC_KEY", 32)
        print(json.dumps(create_checkpoint(
            args.db, args.checkpoint, gateway_key=hmac_key,
            anchor_key=independent_key,
        ), indent=2))
    elif args.command == "verify-anchor":
        independent_key = _load_secret("ARU_ANCHOR_HMAC_KEY", 32)
        result = verify_checkpoint(
            args.db, args.checkpoint, gateway_key=hmac_key,
            anchor_key=independent_key,
        )
        print(json.dumps(result, indent=2))
        if not result["verified"]:
            raise SystemExit(1)
    elif args.command == "report":
        print(json.dumps(build_report_from_gateway(
            args.db, args.out_dir, hmac_key=hmac_key,
            baseline_policy=args.baseline_policy,
            candidate_policy=args.candidate_policy,
            replicates=args.replicates,
        ), indent=2))
    else:
        print(json.dumps(export_jsonl(args.db, args.decisions, args.labels,
                                      hmac_key=hmac_key), indent=2))


if __name__ == "__main__":
    main()
