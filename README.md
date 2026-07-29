# claim_evidence

Turns completed [`document_extract`](../document_extract) output folders into an
evidence-first RAG and a claim-focused knowledge graph.

One question, answered conservatively:

> Given one atomic claim, return `supported`, `contradicted`, `mixed`, or
> `insufficient`, backed by exact source evidence containing the 1-based PDF
> page and one or more bounding boxes.

A claim may be supported by text, tables, or a chart, but **visual evidence must
be re-checked from the cited crop** before it can support a verdict, and
generated Markdown never supports anything on its own.

## Install

```bash
python -m pip install -e .
```

Start the development database (PostgreSQL 18 + pgvector 0.8.5):

```bash
docker compose up -d
```

Pull the models into Ollama:

```bash
ollama pull qwen3-embedding:4b
```

## Configuration

| Variable | Default |
|---|---|
| `CLAIM_EVIDENCE_DATABASE_URL` | `postgresql://claim_evidence:claim_evidence@localhost:5433/claim_evidence` |
| `CLAIM_EVIDENCE_DATABASE_CONNECT_TIMEOUT` | `10` (seconds) |
| `CLAIM_EVIDENCE_BUILD_STALE_MINUTES` | `60` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` |
| `CLAIM_EVIDENCE_EMBED_MODEL` | `qwen3-embedding:4b` |
| `CLAIM_EVIDENCE_EMBED_DIMENSIONS` | `1024` |
| `CLAIM_EVIDENCE_CHAT_MODEL` | `hf.co/unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-Q8_K_XL` |
| `CLAIM_EVIDENCE_VISION_MODEL` | same as the chat model |
| `CLAIM_EVIDENCE_EMBED_BATCH_SIZE` | `32` |
| `CLAIM_EVIDENCE_REQUEST_TIMEOUT` | `600` |

Point `CLAIM_EVIDENCE_DATABASE_URL` at a managed instance to skip Compose
entirely. The embedding dimension is templated into the schema at `db init`, so
changing the model means re-running `db init` on a fresh database.

An unreachable database raises `DependencyUnavailableError` after
`CLAIM_EVIDENCE_DATABASE_CONNECT_TIMEOUT` rather than blocking on the operating
system's network timeout. libpq takes whole seconds and treats anything below 2
as 2. A nonpositive or unparseable value is a `ValidationError` at startup, and
neither error quotes the connection string.

**On the 1024 default:** `qwen3-embedding:4b` natively returns 2560 dimensions
and Ollama's `/api/embed` exposes no output-dimension parameter. Qwen3-Embedding
is Matryoshka-trained, so the client keeps the leading 1024 values and
re-normalizes — the documented way to shorten an MRL embedding. A model that is
*not* MRL-trained must be configured with its native dimension; a vector shorter
than the configured width is a hard error rather than a silent pad.

## CLI

```bash
claim-evidence db init
```

```bash
claim-evidence ingest OUTPUT_ROOT --pdf SOURCE_PDF
```

```bash
claim-evidence search "Scope 1 and 2 emissions versus 2020" --json
```

```bash
claim-evidence audit "Danone reduced Scope 1 and 2 energy and industry emissions by 40.2% in 2025 versus 2020." --json
```

```bash
claim-evidence health
```

```bash
claim-evidence documents
```

```bash
claim-evidence trace AUDIT_ID --json
```

```bash
claim-evidence evidence EVIDENCE_ID --json
```

```bash
claim-evidence remove DOCUMENT_ID --confirm DOCUMENT_ID
```

`ingest --no-narrative-facts` indexes evidence and deterministic table facts but
skips the LLM narrative fact extractor, which is the slow part of a first run.
`ingest --force` rebuilds an unchanged source without taking the current
version out of service.

## Python API

```python
from claim_evidence import ClaimEvidence

client = ClaimEvidence.from_env()

ingest_report = client.ingest_document(
    output_root,
    source_pdf=source_pdf,
    source_uri=None,
    progress=lambda event: print(event.phase, event.percent),  # optional
)

result = client.audit_claim(
    "Danone reduced Scope 1 and 2 energy and industry emissions "
    "by 40.2% in 2025 versus 2020."
)

matches = client.search_evidence("Scope 1 and 2 emissions versus 2020", limit=20)
```

`document_ids` is validated before anything is embedded, parsed, or persisted.
`None` and `[]` both mean every ready document; an unknown id raises
`NotFoundError`, an id with no ready version raises `IndexNotReadyError`, and a
malformed one raises `ValidationError`. A mixed selection fails as a whole
rather than quietly searching the valid part, and an empty index raises rather
than returning an `insufficient` verdict — "nothing was retrieved" is a
statement about the request, not about the report.

`ClaimResult.evidence_quality` is one of `direct_text`, `direct_table`,
`verified_visual`, `coarse_region`, `none`. There is deliberately no numeric
model confidence: an uncalibrated number reads as precision the system does not
have.

### Why a verdict came out that way

`ClaimResult` and `AuditTrace` both carry `decision_explanation`, `timings`, and
`index_references`. The explanation is produced by the same comparison that
decided the verdict, so the two can never disagree, and a caller never has to
re-derive whether a qualifier matched.

```text
decision_explanation.decided_by    deterministic_comparison | semantic_adjudication | no_evidence
decision_explanation.verdict_rule  exact_numeric_match | bounded_numeric_match
                                   | comparable_numeric_conflict | mixed_comparable_facts
                                   | semantic_evidence_support | semantic_evidence_conflict
                                   | scope_not_comparable | no_citable_evidence
                                   | missing_material_qualifier
decision_explanation.evidence_comparisons[]
    evidence_id  fact_id  pdf_page
    qualifiers[] qualifier claim_value source_value status reason
    numeric      claim_value claim_operator claim_direction
                 source_value source_operator source_unit outcome reason
```

A qualifier is `match` only where the comparison actually established
comparability; a qualifier the source omits is `missing`, and `mismatch` needs
both sides present and disagreeing. `numeric.outcome` is `incomparable` when a
qualifier blocked the arithmetic and `not_applicable` when either side states no
number — a broad claim is never turned into a numeric conflict against narrower
evidence. There is one entry per fact examined, so multiple or disagreeing facts
keep their own comparisons.

`verdict_rule` is operational metadata, not model reasoning; the user-facing
prose stays in `rationale`. Nothing in the explanation is a prompt, a raw model
reply, or a local path.

`timings` reports elapsed seconds for `parsing`, `retrieval`, `fusion_context`,
`visual_verification`, `verdict`, `persistence`, and `total`. A group whose
phases never ran is `null` rather than `0.0`. `index_references` pins the exact
ready `document_version_id`, embedding model, and dimension that answered the
audit, and is recorded even for an `insufficient` verdict with no citations.

Each `Citation` carries the document identity, `pdf_page` (always the 1-based
PDF index, never a printed footer number), the quote or table cells, the
artifact it came from, and a list of `Region`s normalized to
`[left, top, right, bottom]` in a top-left 0-1 space with the original
coordinates preserved.

## What the package indexes

The run-level `manifest.json` is the authoritative page list. Stale `page_*`
directories are reported and ignored; any page that is not `completed`, any
duplicate or missing page, and any missing artifact aborts ingestion rather than
producing an index with quiet holes in it.

A run does not have to start at page 1. `page_state.total_pages` is the count of
*selected* pages, so the manifest must cover a contiguous run of that length
starting at its own first page — `1..494`, `1..20`, `10..20`, and a lone `359`
are all valid, and a hole inside the range still fails closed. Page numbers are
never renumbered: PDF page 359 is `359` in evidence, search results, and
citations. Every selected page's checkpoint must agree on the count.

Each manifest `page_dir` is resolved and required to stay strictly inside the
registered output root before any artifact is opened. An absolute path, a `..`
traversal, or a symlink pointing elsewhere is rejected, and the error names the
page rather than the path it tried to reach.

| Artifact | Used for | Citable |
|---|---|---|
| `blocks.jsonl` | headings, prose, block geometry | yes |
| `table_candidates.json` | rows, values, cell geometry | yes |
| `image_summaries.jsonl` | visual retrieval hints | only after crop verification |
| `docling_final.md` | page context | no |
| `page.png` | crops for visual re-verification | via the crop |

Block text prefers `text_full` and falls back to the 500-character `text`,
flagging the unit as truncated. Older runs without `text_full` therefore index
fine, but a strict supporting verdict still needs citable source evidence.

Table values cite four regions -- descriptor, column header, unit, and value
cell. A cell with no geometry of its own falls back to the union of its row's
cells (`row` precision), then to the table (`table` precision).

Table-value cells are not embedded: they are reached through their row, their
fact, and lexical search, and embedding thousands of near-identical `(40.2) %`
strings buys no semantic recall. Narrative blocks, table rows, visual regions,
and page Markdown are embedded.

## How a verdict is reached

1. Parse the claim into subject, metric, value, unit, periods, scope, geography.
2. Retrieve independently through fact/graph filters, PostgreSQL full-text, and
   pgvector.
3. Merge with reciprocal-rank fusion, boosting exact numbers, years, units, and
   scope tokens.
4. Expand the top candidates with their page neighbours.
5. Crop and re-verify any visual candidate.
6. Compare arithmetically whenever every material qualifier aligns. Exact claims
   need exact displayed agreement; "about"/"roughly"/"approximately" allow 5%
   relative tolerance.
7. Fall back to the structured LLM verifier only for semantic qualification and
   ambiguity.

`compare()` returns *incomparable*, not *contradicted*, when a qualifier does
not line up, which is what stops a vague claim from being forced into a
confident contradiction against the nearest number that happens to exist.

Invalid model JSON gets one retry, then a hard error. A supported verdict
without a direct or re-verified citation is downgraded to `insufficient`.

Facts come from two places. Table facts are arithmetic and need no model: the
column header gives the reporting period, `vs. 2020` in the row descriptor
gives the baseline, and accounting parentheses give the sign. Narrative facts go
through the LLM, but each one must echo a quote that really appears in its own
evidence unit or it is discarded.

## Measured on the Danone URD

The completed 494-page run indexes to 21,988 evidence units (5,901 narrative,
3,457 table rows, 11,535 table values, 602 visual, 493 page-Markdown) and
11,168 deterministic table facts, in roughly seven minutes end to end.

Geometry precision across those units: 14,323 cell, 5,901 block, 602 crop, 493
page, 472 row-fallback, 197 table-fallback. The 472 row fallbacks are exactly
the table cells the extractor emitted without their own bounding box.

Scanning every table fact in the document against the three reference claims
yields 2 matches and 0 conflicts for the 40.2% claim, 0 and 2 for the 90%
variant, and nothing comparable for "all carbon emissions" — so the first two
verdicts are decided arithmetically and the third correctly falls through to
`insufficient`.

## Frontend-support API

A local frontend must never query these tables or re-derive the provenance
rules. Everything it needs is on the client:

```python
client.initialize_database()          # idempotent schema + health report
client.health()                       # database, schema, pgvector, models, counts


client.list_documents()               # -> list[DocumentSummary]
client.get_document(document_id)
client.remove_document(document_id, confirm_document_id=document_id)

client.get_audit_trace(audit_id)      # -> AuditTrace
client.get_evidence(evidence_id)      # -> EvidenceDetail
```

Ids may be passed as `int` or `str`; anything else raises `ValidationError`.

### Source identity

Three separate values, because one hash cannot honestly do three jobs:

| Value | What it is |
|---|---|
| `source_sha256` | the actual SHA-256 of the PDF bytes, verifiable with `sha256sum`. Null when no PDF was supplied — never an output fingerprint wearing the source's name. |
| extraction fingerprint | content hash of the manifest, `blocks.jsonl`, and every manifest-listed page's evidence artifacts *including* `page.png`, streamed in page order. |
| version fingerprint | the tagged combination of source hash, extraction fingerprint, embedding model, and dimensions. |

The extraction fingerprint hashes contents, not file sizes. A re-run that
improves a table reconstruction or redraws a page image without changing its
length must not look like no change at all — a same-size edit moves it, and a
stale directory the manifest no longer lists does not.

A better extraction of the same PDF therefore builds a *new version of the same
document*; the previous one is retired, not deleted. `identity_key` is
deliberately still derived from the pre-correction PDF token, so fixing the
public hash does not split every already-indexed PDF into a second document.

`source_uri` is where the report was published. It stays null unless a caller
supplies one; an output directory is not a provenance claim.

### Build states and health totals

A version is `building`, `ready`, `inactive`, or `failed`. When ingestion
raises, that building version is marked `failed` with a safe `failure_code` and
`failure_phase` — never `str(exc)` — and an older `ready` version is left
exactly as it was. Retrying reopens the same failed attempt and clears its
failure metadata first, so a live build is never described by last time's
verdict.

A killed process cannot write its own failure, so `health()` classifies it by
silence instead: still `building` with no progress for longer than
`CLAIM_EVIDENCE_BUILD_STALE_MINUTES` is reported as `documents_interrupted`.
That is a read — `health()` never mutates a row.

```text
documents_ready  documents_building  documents_failed
documents_interrupted  documents_inactive
```

`evidence_units`, `embeddings`, and `facts` count only rows belonging to a
`ready` version: that is the queryable index, and a total that grew when a build
broke would describe storage instead. `stored_evidence_units` keeps the
historical figure for diagnostics.

`schema_embedding_dimensions` and `configured_embedding_dimensions` are both
reported. `vector(N)` is templated in only when the table is first created, so
re-initializing an existing database at a different dimension would otherwise
leave the old column in place; a mismatch fails `db init` with
`IndexNotReadyError` before the first embedding write and makes `health()`
not-current while PostgreSQL stays reachable.

### Errors

`NotFoundError`, `ValidationError`, `DependencyUnavailableError`, and
`IndexNotReadyError` all derive from `ClaimEvidenceError`, so a caller maps a
type rather than parsing a message. Driver exceptions and Ollama response
bodies stay inside the package: `health()` reports a failing query by class,
because a driver message can carry the host and user.

### Re-indexing and removal

`ingest_document(..., force=True)` rebuilds an unchanged source. The current
version keeps serving queries throughout and is only retired once the
replacement passes its integrity checks; if the rebuild fails, the previous
version is still `ready`.

Documents are identified internally by the strongest identity available — the
source PDF's content hash, else `source_uri`, else the canonical output root —
so two unrelated reports whose directories happen to share a basename are two
documents, and removing one cannot remove the other. Registering two copies
under one explicit `source_uri` still makes them versions of one document. The
key is internal: no public result field exposes it.

Resuming an interrupted build reconciles the half-finished version against the
source rather than adding to it. Changed text is overwritten and its embedding
is discarded and recomputed; units, pages, facts, and fact-evidence links the
source no longer produces are deleted. Only the `building` version is ever
touched — a `ready` or `inactive` one is left exactly as it is.

`remove_document()` requires the id twice and deletes only index rows,
returning per-table counts. It never touches the source PDF, the
`document_extract` output directory, page images, or Ollama models —
re-ingestion is the whole recovery path.

### Rendering evidence

`EvidenceDetail` carries `page_image_path` (the registered `page.png`, resolved
from stored metadata — never from a caller-supplied path), `page_width`,
`page_height`, `artifact_paths`, and every `Region`. Regions keep their roles
separate: a table citation is descriptor + header + unit + value, not one box
around the lot.

```text
claim_text  descriptor  header  unit  value  supporting_context  visual_region  unknown
```

Each region's `bbox` is `[left, top, right, bottom]` with
`coordinate_space = "normalized_top_left"`, so multiplying by the page image's
pixel size is all a renderer has to do. The package deliberately draws nothing:

```python
detail = client.get_evidence(evidence_id)
page = Image.open(detail.page_image_path)
w, h = page.size
for region in detail.regions:
    left, top, right, bottom = region.bbox
    draw.rectangle((left * w, top * h, right * w, bottom * h), outline=colors[region.role])
```

`output_root` and `source_pdf` on `DocumentSummary` are server-side path
metadata. A frontend backend may use them after its own allowed-root
validation; they are not for an untrusted browser.

### Progress events

Both long operations accept an optional callback. Callers passing nothing are
unaffected.

```python
client.ingest_document(output_root, progress=events.append)
client.audit_claim(claim, progress=events.append)
```

Each `ProgressEvent` carries `operation`, `phase`, `status`
(`start`/`progress`/`completed`/`warning`/`failed`), a human-readable
`message`, `completed`/`total`/`percent`, `document_id`, `audit_id`,
`current_item`, `details`, and `timestamp`.

`completed`, `total`, and `percent` are null when the work genuinely cannot be
measured — an honest spinner beats an invented percentage. `percent` is
phase-local and never decreases within a phase. A phase with nothing to do
reports `0/0` at 100%, not a division error.

Phases, in order:

```text
ingest   validating_input building_evidence embedding_evidence extracting_facts
         building_indexes activating_version completed
audit    parsing_claim retrieving_graph retrieving_full_text retrieving_vectors
         fusing_candidates expanding_context verifying_visuals deciding_verdict
         persisting_trace completed
```

Totals are real: pages for evidence, embedding batches, claim-like passages for
fact extraction, integrity steps for index checks, and returned-versus-requested
candidates for each retrieval channel.

A callback that raises is dropped for the rest of the operation and the work
continues. A broken UI must never corrupt an index build.

#### Completion summaries

The terminal `completed` event carries counts the operation already computed —
no extra queries. Ingestion mirrors its `IngestReport`:

```text
document_version_id page_count evidence_count visual_evidence_count
embedding_count fact_count warning_count elapsed_seconds no_op
```

Audit reports its retrieval story:

```text
verdict citation_count graph_candidate_count full_text_candidate_count
vector_candidate_count fused_candidate_count expanded_candidate_count
visual_candidate_count visually_verified_count selected_evidence_count
elapsed_seconds
```

Channel counts are what each retriever *returned*; the persisted trace records
only the candidates that survived fusion's cut, so the event count is the
larger of the two. A channel that could not run at all — for example vectors
when embeddings are unavailable — omits its key rather than reporting zero,
because "ran and found nothing" and "never ran" are different facts.

#### Failures

A terminal `failed` event carries `failed_phase`, `error_code`, and `retryable`
in `details`, then the original exception is re-raised unchanged.

| `error_code` | `retryable` |
|---|---|
| `dependency_unavailable` | yes |
| `internal_error` | yes |
| `index_not_ready` | yes |
| `validation_error` | no |
| `not_found` | no |

Only the package's own typed errors are quoted back to the caller; their
messages are written for public consumption. Anything else — a driver error, a
model response body, an unexpected bug — is reported by category, because those
messages carry hosts, credentials, prompts, and source text.

### Retrieval trace

`get_audit_trace()` returns the parsed claim, every candidate with its
per-channel rank and score, the fused rank and score, context-expansion links,
visual re-verification status, whether it was selected, and a short reason.
This is retrieval metadata, not model reasoning — no prompts and no
chain-of-thought are stored.

## Tests

```bash
python tests/run_all.py
```

```bash
python -m compileall -q src
```

The deterministic suites need no GPU, no Ollama, and no Docker; Ollama is
replaced by a fake HTTP session. `tests/test_integration.py` needs the Compose
database and skips cleanly without it.

The live acceptance run is separate, because a first ingest of a 494-page
report is not a unit test:

```bash
python tests/acceptance_danone.py
```

It indexes the completed Danone output and audits the three reference claims
(supported at 40.2%, contradicted at 90%, and an "all carbon emissions" claim
that must stay `insufficient`). It skips cleanly when the output root,
database, or models are missing.

## Non-goals

No autonomous agent loop, no REST API or MCP server, no generic ontology or
community detection, no separate graph and vector stores, and one atomic claim
per audit call.
