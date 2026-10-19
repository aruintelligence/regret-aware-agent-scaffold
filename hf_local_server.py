"""Local OpenAI-compatible endpoint backed by an ACTUAL Hugging Face model.

This is a CPU-only, loopback-only research adapter for reproducible evaluation.
It does not start automatically, install a model, or send user data to a
third-party inference API. Model weights must be downloaded separately.
The first download requires network access to the model hosting provider.

Only public, non-sensitive benchmark questions should be sent here.
Not hardened for production, multi-user use, or untrusted public traffic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from grounding_contract import ground_answer, verify_grounding

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
MAX_REQUEST_BYTES = 30000
MAX_NEW_TOKENS = 128


class LocalModelEngine:
    """Loads one fixed pretrained model and generates deterministic responses."""

    def __init__(self, model_id=DEFAULT_MODEL, *, revision=None, threads=2,
                 local_files_only=False):
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("nonempty model identifier required")
        if threads < 1 or threads > 32:
            raise ValueError("threads must be in [1,32]")
        # Optional dependencies are loaded only when explicitly starting the model.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(threads)
        kwargs = {
            "trust_remote_code": False,
            "local_files_only": local_files_only,
        }
        if revision:
            kwargs["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        # Refuse legacy pickle weight formats; prefer audited safetensors.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, use_safetensors=True, **kwargs,
        )
        self.model.eval()
        self.torch = torch
        self.model_id = model_id
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision
        self.lock = threading.Lock()

    def generate(self, messages, *, max_tokens=96):
        if not isinstance(messages, list) or not messages:
            raise ValueError("nonempty message list required")
        if any(not isinstance(m, dict)
               or m.get("role") not in ("system", "user", "assistant")
               or not isinstance(m.get("content"), str) for m in messages):
            raise ValueError("messages must be text with supported roles")
        if sum(len(m["content"]) for m in messages) > 25000:
            raise ValueError("messages too long")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= MAX_NEW_TOKENS:
            raise ValueError("max_tokens must be in [1,128]")
        with self.lock, self.torch.inference_mode():
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
            )
            if input_ids.shape[-1] > 4096:
                raise ValueError("tokenized prompt exceeds 4096 tokens")
            generated = self.model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            new_ids = generated[0, input_ids.shape[-1]:]
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            return {
                "choices": [{"index": 0, "message": {
                    "role": "assistant", "content": text,
                }, "finish_reason": "stop"}],
                "model": self.model_id,
                "usage": {
                    "prompt_tokens": int(input_ids.shape[-1]),
                    "completion_tokens": int(new_ids.shape[-1]),
                },
            }


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ARU-Local-Research-Model/0.1"
        sys_version = ""

        def log_message(self, *args):
            # Never log raw questions, prompts, or model responses.
            pass

        def respond(self, status, body):
            payload = json.dumps(body, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/health":
                self.respond(200, {
                    "status": "ready",
                    "model": engine.model_id,
                    "revision": engine.revision,
                    "scope": "loopback_research_only",
                })
            else:
                self.respond(404, {"error": "not_found"})

        def do_POST(self):
            if self.path not in ("/v1/chat/completions", "/v1/grounded/answers"):
                self.respond(404, {"error": "not_found"})
                return
            grounded = self.path == "/v1/grounded/answers"
            if grounded:
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
            if size < 2 or size > MAX_REQUEST_BYTES:
                self.respond(413, {"error": "invalid_request_size"})
                return
            try:
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict):
                    raise ValueError("object required")
                if request.get("stream", False):
                    raise ValueError("streaming not supported")
                if request.get("temperature", 0) != 0:
                    raise ValueError("only deterministic temperature=0 supported")
                if request.get("model") != engine.model_id:
                    raise ValueError("model identifier does not match loaded model")
                if grounded:
                    if set(request) - {"model", "context", "question", "max_tokens"}:
                        raise ValueError("unexpected grounded request fields")
                    context = request["context"]
                    question = request["question"]
                    if not isinstance(context, str) or not isinstance(question, str):
                        raise ValueError("context and question must be text")
                    if not context or len(context) > 25000 or not question or len(question) > 2000:
                        raise ValueError("context or question length invalid")
                    # Use the same frozen extractive prompt as the real-model benchmark.
                    messages = [
                        {"role": "system", "content":
                         "You are an extractive question-answering engine. "
                         "Find the shortest exact text span from the passage that answers "
                         "the question. Return only that span verbatim. No explanations, "
                         "no invented facts. If absent, return UNKNOWN."},
                        {"role": "user", "content":
                         "Passage:\n" + context + "\n\nQuestion:\n" + question},
                    ]
                    raw = engine.generate(
                        messages, max_tokens=request.get("max_tokens", 48),
                    )
                    model_text = raw["choices"][0]["message"]["content"]
                    proof = ground_answer(model_text, context, model_revision=engine.revision)
                    if proof["accepted"] and not verify_grounding(proof, context):
                        raise RuntimeError("independent source proof check failed")
                    response = {
                        "model": engine.model_id,
                        "accepted": proof["accepted"],
                        "answer": proof["answer"],
                        "proof": proof,
                        "usage": raw.get("usage", {}),
                        "automatic_rollout_permitted": False,
                        "warning": "Source substring provenance only, not factual correctness or entailment.",
                    }
                else:
                    response = engine.generate(
                        request["messages"],
                        max_tokens=request.get("max_tokens", 96),
                    )
            except (KeyError, ValueError, TypeError):
                self.respond(422, {"error": "invalid_model_request"})
                return
            except Exception:
                # Avoid leaking local model or environment internals to callers.
                self.respond(503, {"error": "model_inference_failed"})
                return
            self.respond(200, response)

    return Handler


def make_server(engine, *, host="127.0.0.1", port=1234):
    if host != "127.0.0.1":
        raise ValueError("local model server must bind only to IPv4 loopback")
    return ThreadingHTTPServer((host, port), make_handler(engine))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", help="Optional immutable Hugging Face model commit SHA")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--metadata-json", help="Optional non-secret model metadata file")
    args = parser.parse_args()
    engine = LocalModelEngine(
        args.model, revision=args.revision,
        threads=args.threads, local_files_only=args.local_files_only,
    )
    if args.metadata_json:
        target = Path(args.metadata_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "model": engine.model_id,
            "revision": engine.revision,
            "backend": "transformers_pytorch_cpu",
            "deterministic_decoding": True,
            "remote_code_enabled": False,
            "source": "Hugging Face pretrained model weights, operator supplied",
        }, indent=2) + "\n", encoding="utf-8")
    server = make_server(engine, port=args.port)
    print("Local research model ready on 127.0.0.1:" + str(server.server_port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
