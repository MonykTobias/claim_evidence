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

`ingest --no-narrative-facts` indexes evidence and deterministic table facts but
skips the LLM narrative fact extractor, which is the slow part of a first run.

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

## Non-goals

No autonomous agent loop, no REST API or MCP server, no generic ontology or
community detection, no separate graph and vector stores, and one atomic claim
per audit call.
