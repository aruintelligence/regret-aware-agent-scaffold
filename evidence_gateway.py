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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from evidence_gateway_store import (
    EvidenceConflict, export_jsonl, record_event, verify_chain,
)


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
    if host not in ("127.0.0.1", "::1"):
        raise ValueError("gateway must bind only to loopback")
    return ThreadingHTTPServer((host, port), make_handler(db_path, api_key, hmac_key))


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
    else:
        print(json.dumps(export_jsonl(args.db, args.decisions, args.labels,
                                      hmac_key=hmac_key), indent=2))


if __name__ == "__main__":
    main()
