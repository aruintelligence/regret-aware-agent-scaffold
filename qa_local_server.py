"""Authenticated loopback-only research API for a REAL pretrained extractive QA model.

POST /v1/qa/proof accepts approved source text and a question, returning
a verified source-offset certificate or abstention. No raw prompts are logged.
NOT an internet-facing production service: no TLS, RBAC, tenant isolation,
retention/deletion service, or verified human consent.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from extractive_qa_engine import DEFAULT_QA_MODEL, ExtractiveQAEngine
from grounding_contract import verify_grounding
from sovereign_source_registry import authorize_source
from source_registry_anchor import verify_source_checkpoint
from quorum_registry_adapter import evaluate_registered_quorum

MAX_BODY = 30000


def make_handler(engine, api_key, *, registry_db=None, registry_key=None,
                 registry_checkpoint=None, registry_anchor_key=None):
    if not isinstance(api_key, bytes) or len(api_key) < 24:
        raise ValueError("QA API key must be at least 24 bytes")
    if registry_db is not None and (
        not isinstance(registry_key, bytes) or len(registry_key) < 32
    ):
        raise ValueError("separate 32-byte source registry key required")
    if registry_checkpoint is not None and (
        registry_db is None or not isinstance(registry_anchor_key, bytes)
        or len(registry_anchor_key) < 32
    ):
        raise ValueError("anchored consent requires registry and separate 32-byte anchor key")

    class Handler(BaseHTTPRequestHandler):
        server_version = "ARU-QA-Research/0.1"
        sys_version = ""

        def log_message(self, *args):
            # No URL, source, answer or key logging.
            pass

        def respond(self, status, body):
            raw = json.dumps(body, sort_keys=True, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path != "/health":
                self.respond(404, {"error": "not_found"})
            else:
                self.respond(200, {
                    "status": "ready", "model": engine.model_id,
                    "revision": engine.revision, "scope": "authenticated_loopback_research",
                })

        def do_POST(self):
            if self.path not in ("/v1/qa/proof", "/v1/qa/quorum"):
                self.respond(404, {"error": "not_found"})
                return
            header = self.headers.get("Authorization", "")
            provided = header[7:].encode("utf-8") if header.startswith("Bearer ") else b""
            if not hmac.compare_digest(provided, api_key):
                self.respond(401, {"error": "unauthorized"})
                return
            if self.headers.get("X-ARU-Consent") != "research-opt-in":
                self.respond(403, {"error": "explicit_research_consent_header_required"})
                return
            if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
                self.respond(415, {"error": "application_json_required"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.respond(400, {"error": "invalid_content_length"})
                return
            if not 2 <= size <= MAX_BODY:
                self.respond(413, {"error": "invalid_request_size"})
                return
            try:
                payload = json.loads(self.rfile.read(size))
                if self.path == "/v1/qa/quorum":
                    if not isinstance(payload, dict) or set(payload) - {
                        "model", "question", "sources",
                        "min_distinct_sources", "max_answer_tokens",
                    }:
                        raise ValueError("invalid quorum request fields")
                    if payload.get("model") != engine.model_id:
                        raise ValueError("quorum model does not match loaded model")
                    receipt = evaluate_registered_quorum(
                        engine,
                        registry_db=registry_db, registry_key=registry_key,
                        registry_checkpoint=registry_checkpoint,
                        registry_anchor_key=registry_anchor_key,
                        owner=self.headers.get("X-ARU-Owner", ""),
                        question=payload["question"],
                        sources=payload["sources"],
                        min_distinct_sources=payload.get("min_distinct_sources", 2),
                        max_answer_tokens=payload.get("max_answer_tokens", 30),
                    )
                    self.respond(200, receipt)
                    return
                if not isinstance(payload, dict) or set(payload) - {
                    "model", "context", "question", "max_answer_tokens",
                }:
                    raise ValueError("invalid request fields")
                if payload.get("model") != engine.model_id:
                    raise ValueError("model does not match loaded model")
                context, question = payload["context"], payload["question"]
                max_tokens = payload.get("max_answer_tokens", 30)
                if not isinstance(context, str) or not isinstance(question, str):
                    raise ValueError("context and question must be text")
                if not 0 < len(context) <= 25000 or not 0 < len(question) <= 2000:
                    raise ValueError("invalid context or question length")
                if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 64:
                    raise ValueError("invalid answer token limit")
                source_authorization = None
                if registry_db is not None:
                    if registry_checkpoint is not None:
                        try:
                            checkpoint_result = verify_source_checkpoint(
                                registry_db, registry_checkpoint,
                                registry_key=registry_key,
                                anchor_key=registry_anchor_key,
                            )
                        except (ValueError, OSError):
                            self.respond(403, {"error": "source_checkpoint_unavailable"})
                            return
                        if not checkpoint_result["verified"]:
                            self.respond(403, {"error": "source_checkpoint_invalid"})
                            return
                    owner = self.headers.get("X-ARU-Owner", "")
                    if not owner:
                        self.respond(403, {"error": "source_owner_required"})
                        return
                    source_authorization = authorize_source(
                        registry_db, context, owner=owner, purpose="qa_source",
                        key=registry_key,
                    )
                    if not source_authorization["authorized"]:
                        self.respond(403, {
                            "error": "source_not_authorized",
                            "reason": source_authorization["reason"],
                        })
                        return
                answer = engine.answer(
                    context, question, max_answer_tokens=max_tokens,
                )
                if answer["accepted"] and not verify_grounding(answer["proof"], context):
                    raise RuntimeError("independent source verification failed")
                self.respond(200, {
                    "accepted": answer["accepted"],
                    "answer": answer["answer"],
                    "proof": answer["proof"],
                    "model": engine.model_id,
                    "model_revision": engine.revision,
                    "reason": answer["reason"],
                    "source_authorization": {
                        "kind": "operator_attested_purpose_scoped_registry",
                        "grant_id": source_authorization["grant_id"],
                        "purpose": "qa_source",
                        "independent_checkpoint_verified": registry_checkpoint is not None,
                        "human_identity_verified": False,
                        "legal_consent_verified": False,
                        "raw_source_stored_by_registry": False,
                    } if source_authorization else {
                        "kind": "research_header_only_no_verified_consent",
                        "human_identity_verified": False,
                        "legal_consent_verified": False,
                    },
                    "automatic_rollout_permitted": False,
                    "warning": "Exact source-span provenance only; not factual correctness or entailment.",
                })
            except PermissionError:
                self.respond(403, {"error": "source_not_authorized_or_revoked"})
            except (ValueError, KeyError, TypeError):
                self.respond(422, {"error": "invalid_request"})
            except Exception:
                self.respond(503, {"error": "qa_inference_unavailable"})

    return Handler


def make_server(engine, api_key, *, host="127.0.0.1", port=1235,
                registry_db=None, registry_key=None,
                registry_checkpoint=None, registry_anchor_key=None):
    if host != "127.0.0.1":
        raise ValueError("QA research server must bind only to IPv4 loopback")
    return ThreadingHTTPServer((host, port), make_handler(
        engine, api_key, registry_db=registry_db, registry_key=registry_key,
        registry_checkpoint=registry_checkpoint,
        registry_anchor_key=registry_anchor_key,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_QA_MODEL)
    parser.add_argument("--revision", help="Immutable model revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--port", type=int, default=1235)
    parser.add_argument("--source-registry-db",
                        help="Optional consent-scoped source ledger; deny by default")
    parser.add_argument("--source-anchor-checkpoint",
                        help="Optional independently retained HMAC checkpoint for ledger prefix")
    args = parser.parse_args()
    key = os.environ.get("ARU_QA_API_KEY", "").encode("utf-8")
    if len(key) < 24:
        parser.error("ARU_QA_API_KEY must contain at least 24 bytes")
    engine = ExtractiveQAEngine(
        args.model, revision=args.revision,
        local_files_only=args.local_files_only, threads=args.threads,
    )
    registry_key = None
    if args.source_registry_db:
        registry_key = os.environ.get("ARU_SOURCE_REGISTRY_KEY", "").encode("utf-8")
        if len(registry_key) < 32:
            parser.error("ARU_SOURCE_REGISTRY_KEY must contain at least 32 bytes")
    registry_anchor_key = None
    if args.source_anchor_checkpoint:
        if not args.source_registry_db:
            parser.error("--source-anchor-checkpoint requires --source-registry-db")
        registry_anchor_key = os.environ.get("ARU_SOURCE_ANCHOR_KEY", "").encode("utf-8")
        if len(registry_anchor_key) < 32:
            parser.error("ARU_SOURCE_ANCHOR_KEY must contain a separate 32+ byte secret")
    server = make_server(
        engine, key, port=args.port,
        registry_db=args.source_registry_db, registry_key=registry_key,
        registry_checkpoint=args.source_anchor_checkpoint,
        registry_anchor_key=registry_anchor_key,
    )
    print("Authenticated QA research server ready on IPv4 loopback", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
