# ĀRU Sovereign Source Registry — Consent-Scoped Research Memory

**Status:** implemented local research control, not a production consent-management or identity-verification service.

## Why this matters

A source-span certificate proves that an answer was extracted from some supplied text. It does **not** prove that the assistant had permission to use that text. ĀRU's remembrance-first principles require a second question: **was this specific source authorized for this purpose at the time it was used?**

The source registry provides an explicit, bounded answer. It stores an HMAC-keyed fingerprint of the document and an opaque owner identifier, an allowed purpose, an expiration, and a revocable grant. It does **not** store the raw source text, question, answer, or owner identifier.

## Key properties

| Property | Implemented behavior |
| --- | --- |
| Default-deny | Missing registry, missing grant, altered source, wrong owner or wrong purpose are denied |
| Explicit operator attestation | A grant requires `--consent-asserted`; this is **not independently verified human consent** |
| Purpose limitation | A grant for `qa_source` does not permit `memory_recall` |
| Expiry | Grants last at most 30 days; authorization checks the current time |
| Revocation | A matching owner can append a revocation; future requests are denied |
| Minimized storage | Keyed owner/source fingerprints; no raw source, prompt, answer or identity |
| Integrity | HMAC-chained events; missing internal events and tampering are detected |
| Independent access check | The local QA server can require an active grant before model execution |
| Fail closed | Invalid registry signatures, missing owner and revoked/expired grants deny use |

## Local setup

Use separate, strong secrets. Keep them outside source control.

```bash
export ARU_QA_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
export ARU_SOURCE_REGISTRY_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"

# Create a document file and an opaque, non-PII owner identifier file.
# Obtain and verify actual permission using a process OUTSIDE this prototype.
aru-source-registry grant \
  --db private-research/consent.db \
  --source-file approved-passage.txt \
  --owner-file opaque-owner.txt \
  --purpose qa_source --ttl-seconds 3600 \
  --consent-asserted

aru-source-registry check \
  --db private-research/consent.db \
  --source-file approved-passage.txt \
  --owner-file opaque-owner.txt \
  --purpose qa_source

aru-source-registry verify --db private-research/consent.db

# Start the actual pretrained extractive QA model with mandatory source grants.
aru-qa-serve --source-registry-db private-research/consent.db
```

For `POST /v1/qa/proof`, send the usual `Authorization: Bearer <ARU_QA_API_KEY>` and `X-ARU-Consent: research-opt-in` headers, **plus `X-ARU-Owner: <opaque-owner-id>`**. The exact source passage in the request must match the active `qa_source` grant. The API checks authorization **before running the model**.

To revoke:

```bash
aru-source-registry revoke --db private-research/consent.db \
  --grant-id <grant-id-from-grant-response> \
  --owner-file opaque-owner.txt
```

A revoked or expired grant cannot authorize subsequent requests. A new grant requires another explicit operator attestation.

## Security boundaries and important non-claims

1. **No verified user identity.** The owner header is caller-supplied. A bearer-token holder can impersonate another owner within this single-operator research environment. Production requires authenticated user identities and binding grants to that identity.
2. **No independently verified consent.** The CLI records an operator's assertion. Actual legal consent, source ownership and data-processing rights must be checked separately.
3. **No encrypted source storage because no source is stored.** The SQLite database still holds metadata and is not encrypted at rest. Keyed fingerprints reduce exposure but do not remove metadata risks.
4. **No tail-deletion resistance without external anchoring.** A privileged actor may delete the final HMAC events and leave a valid shorter prefix. An independently retained checkpoint or external append-only transparency service is required before claiming durable revocation evidence.
5. **No instant in-flight cancellation.** Revocation blocks requests that check authorization afterward; it does not cancel an inference already running.
6. **No multi-tenancy or public deployment.** Local loopback, bearer key and consent header are research controls, not production SSO, RBAC, TLS, verified consent, or tenant isolation.
7. **No factual guarantee.** A permitted source can be inaccurate, malicious or irrelevant; a valid source-span certificate proves provenance only.

## Tests and falsification

`tests/test_sovereign_source_registry.py` tests default-deny, explicit attestation, no raw source storage, exact source/owner/purpose matching, expiration, revocation, regrant, HMAC tampering and missing internal events. `tests/test_qa_local_server.py` tests authorization **before** model execution, wrong-owner and modified-source denial, and immediate denial after revocation. `tests/test_source_registry_cli.py` tests the complete operator workflow.

**Product principle:** ĀRU should not merely remember an answer. It should be able to explain *which authorized source it used, for what purpose, and whether permission remains valid*.
