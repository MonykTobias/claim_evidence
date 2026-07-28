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

ingest_report = client.ingest_document(output_root, source_pdf=source_pdf, source_uri=None)

result = client.audit_claim(
    "Danone reduced Scope 1 and 2 energy and industry emissions "
    "by 40.2% in 2025 versus 2020."
)

matches = client.search_evidence("Scope 1 and 2 emissions versus 2020", limit=20)
```

`ClaimResult.evidence_quality` is one of `direct_text`, `direct_table`,
`verified_visual`, `coarse_region`, `none`. There is deliberately no numeric
model confidence: an uncalibrated number reads as precision the system does not
have.

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
