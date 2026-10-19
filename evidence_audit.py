"""Optional evidence bundle checksums and HMAC authentication.

SHA-256 alone detects changes against a trusted external copy of the manifest.
HMAC requires a separately protected secret and does not make local JSONL
immutable. Store manifests externally for an auditable chain of custody.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from evidence_lab import load_labels
from route_experiments import load_decisions


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(path: str) -> dict:
    file = Path(path)
    hasher = hashlib.sha256()
    with file.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            hasher.update(block)
    return {"filename": file.name, "bytes": file.stat().st_size,
            "sha256": hasher.hexdigest()}


def build_manifest(decisions_path: str, labels_path: str,
                   report_path: Optional[str] = None,
                   secret: Optional[bytes] = None) -> dict:
    decisions = load_decisions(decisions_path)
    labels = load_labels(labels_path)
    decision_ids = {row["decision_id"] for row in decisions}
    if set(labels) - decision_ids:
        raise ValueError("labels include unknown decision IDs")
    files = {
        "decisions": _digest(decisions_path),
        "labels": _digest(labels_path),
    }
    if report_path:
        files["report"] = _digest(report_path)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "sha256",
        "counts": {"decisions": len(decisions), "labels": len(labels)},
        "files": files,
        "authentication": "hmac-sha256" if secret is not None else "unsigned-checksums",
    }
    if secret is not None:
        if not isinstance(secret, bytes) or len(secret) < 16:
            raise ValueError("HMAC secret must be at least 16 bytes")
        manifest["hmac_sha256"] = hmac.new(secret, _canonical(manifest), hashlib.sha256).hexdigest()
    return manifest


def save_manifest(manifest: dict, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite an existing attestation without explicit operator action.
    with target.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def verify_manifest(manifest_path: str, decisions_path: str, labels_path: str,
                    report_path: Optional[str] = None,
                    secret: Optional[bytes] = None) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema")
    errors = []
    authentication = manifest.get("authentication")
    if authentication == "hmac-sha256":
        signature = manifest.get("hmac_sha256")
        if not secret:
            errors.append("HMAC key required to verify authenticated manifest")
        elif not isinstance(signature, str) or len(signature) != 64:
            errors.append("Invalid or missing HMAC signature")
        else:
            payload = {key: value for key, value in manifest.items() if key != "hmac_sha256"}
            expected = hmac.new(secret, _canonical(payload), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                errors.append("Manifest HMAC authentication failed")
    elif authentication == "unsigned-checksums":
        if secret is not None:
            errors.append("Authentication downgrade: HMAC key supplied but manifest is unsigned")
    else:
        errors.append("Unsupported authentication mode")
    paths = {"decisions": decisions_path, "labels": labels_path}
    if report_path:
        paths["report"] = report_path
    for role, expected in manifest.get("files", {}).items():
        if role not in paths:
            errors.append(f"Missing path for {role}")
            continue
        try:
            actual = _digest(paths[role])
        except OSError:
            errors.append(f"Cannot read {role} file")
            continue
        if actual["sha256"] != expected.get("sha256") or actual["bytes"] != expected.get("bytes"):
            errors.append(f"File changed: {role}")
    if set(paths) != set(manifest.get("files", {})):
        errors.append("Manifest file roles do not match provided inputs")
    return {
        "verified": not errors,
        "authentication": authentication,
        "errors": errors,
        "warning": "Unsigned checksums require a separately trusted manifest"
                   if authentication == "unsigned-checksums" else
                   "HMAC proves key possession, not the truth of labels or the immutability of storage",
    }
