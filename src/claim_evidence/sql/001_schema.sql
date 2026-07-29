-- claim_evidence schema.
--
-- Re-runnable: every object is created IF NOT EXISTS so `db init` is safe on a
-- populated database. `EMBED_DIM` is substituted with the configured embedding
-- dimension before execution.

CREATE EXTENSION IF NOT EXISTS vector;

-- Lets health() report whether an existing database has the current shape.
CREATE TABLE IF NOT EXISTS schema_meta (
    id         integer     PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    version    integer     NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document (
    id           bigserial PRIMARY KEY,
    name         text        NOT NULL,
    sha256       text,
    source_uri   text,
    -- Internal identity only, never a public API field. (name, sha256) was not
    -- an identity: two unrelated output roots that happen to share a basename
    -- both key as (same_name, NULL) and collapse into one document. This is
    -- sha256 of a namespace-tagged basis -- the PDF hash, else the logical
    -- source URI, else the canonical output root -- so equal text in different
    -- namespaces cannot collide. Nullable here, made NOT NULL after the
    -- backfill at the end of this file.
    identity_key text,
    created_at   timestamptz NOT NULL DEFAULT now()
);

-- A version is only visible to queries once it reaches 'ready'. An interrupted
-- build stays 'building' forever and is simply never joined against.
CREATE TABLE IF NOT EXISTS document_version (
    id            bigserial PRIMARY KEY,
    document_id   bigint      NOT NULL REFERENCES document(id) ON DELETE CASCADE,
    fingerprint   text        NOT NULL,
    embed_model   text        NOT NULL,
    embed_dim     integer     NOT NULL,
    status        text        NOT NULL DEFAULT 'building'
                  CHECK (status IN ('building', 'ready', 'inactive')),
    output_root   text        NOT NULL,
    source_pdf    text,
    -- A forced rebuild re-indexes an unchanged source, so identity is
    -- (document, fingerprint, attempt): the new 'building' row coexists with
    -- the old 'ready' one, which keeps serving queries until the swap.
    attempt       integer     NOT NULL DEFAULT 1,
    created_at    timestamptz NOT NULL DEFAULT now(),
    ready_at      timestamptz,
    UNIQUE (document_id, fingerprint, attempt)
);

CREATE INDEX IF NOT EXISTS document_version_ready_idx
    ON document_version (document_id, status);

CREATE TABLE IF NOT EXISTS page (
    id                  bigserial PRIMARY KEY,
    version_id          bigint  NOT NULL REFERENCES document_version(id) ON DELETE CASCADE,
    pdf_page            integer NOT NULL,
    printed_page_label  text,
    width               double precision NOT NULL,
    height              double precision NOT NULL,
    page_dir            text    NOT NULL,
    UNIQUE (version_id, pdf_page)
);

CREATE TABLE IF NOT EXISTS evidence_unit (
    id                 bigserial PRIMARY KEY,
    version_id         bigint  NOT NULL REFERENCES document_version(id) ON DELETE CASCADE,
    page_id            bigint  NOT NULL REFERENCES page(id) ON DELETE CASCADE,
    unit_key           text    NOT NULL,
    kind               text    NOT NULL,
    quality            text    NOT NULL,
    citable            boolean NOT NULL DEFAULT true,
    source_text        text    NOT NULL,
    normalized_text    text    NOT NULL,
    heading_path       jsonb   NOT NULL DEFAULT '[]'::jsonb,
    table_context      jsonb   NOT NULL DEFAULT '{}'::jsonb,
    artifact_path      text    NOT NULL,
    geometry_precision text    NOT NULL,
    truncated_source   boolean NOT NULL DEFAULT false,
    embedding          vector(EMBED_DIM),
    text_search        tsvector GENERATED ALWAYS AS
                       (to_tsvector('english', normalized_text)) STORED,
    UNIQUE (version_id, unit_key)
);

CREATE INDEX IF NOT EXISTS evidence_unit_fts_idx
    ON evidence_unit USING gin (text_search);

-- Embeddings are L2-normalized on write, so inner product ranks identically to
-- cosine similarity while staying the cheaper operator.
CREATE INDEX IF NOT EXISTS evidence_unit_embedding_idx
    ON evidence_unit USING hnsw (embedding vector_ip_ops);

CREATE INDEX IF NOT EXISTS evidence_unit_version_idx ON evidence_unit (version_id);
CREATE INDEX IF NOT EXISTS evidence_unit_page_idx    ON evidence_unit (page_id);
CREATE INDEX IF NOT EXISTS evidence_unit_kind_idx    ON evidence_unit (version_id, kind);

CREATE TABLE IF NOT EXISTS evidence_region (
    id            bigserial PRIMARY KEY,
    evidence_id   bigint  NOT NULL REFERENCES evidence_unit(id) ON DELETE CASCADE,
    ordinal       integer NOT NULL,
    role          text    NOT NULL,
    precision     text    NOT NULL,
    left_norm     double precision NOT NULL,
    top_norm      double precision NOT NULL,
    right_norm    double precision NOT NULL,
    bottom_norm   double precision NOT NULL,
    source_bbox   jsonb,
    source_origin text,
    UNIQUE (evidence_id, ordinal)
);

CREATE INDEX IF NOT EXISTS evidence_region_evidence_idx ON evidence_region (evidence_id);

CREATE TABLE IF NOT EXISTS entity (
    id              bigserial PRIMARY KEY,
    kind            text NOT NULL CHECK (kind IN ('organization', 'metric', 'topic', 'geography')),
    canonical_name  text NOT NULL,
    normalized_name text NOT NULL,
    UNIQUE (kind, normalized_name)
);

CREATE TABLE IF NOT EXISTS entity_alias (
    id              bigserial PRIMARY KEY,
    entity_id       bigint NOT NULL REFERENCES entity(id) ON DELETE CASCADE,
    alias           text   NOT NULL,
    normalized_alias text  NOT NULL,
    UNIQUE (entity_id, normalized_alias)
);

CREATE INDEX IF NOT EXISTS entity_alias_lookup_idx ON entity_alias (normalized_alias);

CREATE TABLE IF NOT EXISTS fact (
    id                bigserial PRIMARY KEY,
    version_id        bigint  NOT NULL REFERENCES document_version(id) ON DELETE CASCADE,
    fact_key          text    NOT NULL,
    subject_entity_id bigint  REFERENCES entity(id) ON DELETE SET NULL,
    subject           text    NOT NULL,
    metric            text    NOT NULL,
    normalized_metric text    NOT NULL,
    value_decimal     numeric,
    value_text        text,
    unit              text,
    direction         text    NOT NULL DEFAULT 'unknown',
    comparison        text    NOT NULL DEFAULT '=',
    reporting_period  text,
    baseline_period   text,
    scope             text,
    geography         text,
    qualifiers        jsonb   NOT NULL DEFAULT '{}'::jsonb,
    extraction_method text    NOT NULL,
    quote             text    NOT NULL DEFAULT '',
    UNIQUE (version_id, fact_key)
);

CREATE INDEX IF NOT EXISTS fact_metric_idx  ON fact (version_id, normalized_metric);
CREATE INDEX IF NOT EXISTS fact_period_idx  ON fact (version_id, reporting_period, baseline_period);
CREATE INDEX IF NOT EXISTS fact_scope_idx   ON fact (version_id, scope);
CREATE INDEX IF NOT EXISTS fact_subject_idx ON fact (version_id, subject_entity_id);

CREATE TABLE IF NOT EXISTS fact_evidence (
    fact_id     bigint NOT NULL REFERENCES fact(id) ON DELETE CASCADE,
    evidence_id bigint NOT NULL REFERENCES evidence_unit(id) ON DELETE CASCADE,
    PRIMARY KEY (fact_id, evidence_id)
);

CREATE INDEX IF NOT EXISTS fact_evidence_evidence_idx ON fact_evidence (evidence_id);

CREATE TABLE IF NOT EXISTS audit_run (
    id                 bigserial PRIMARY KEY,
    claim              text        NOT NULL,
    parsed_claim       jsonb       NOT NULL DEFAULT '{}'::jsonb,
    verdict            text,
    rationale          text,
    evidence_quality   text,
    missing_qualifiers jsonb       NOT NULL DEFAULT '[]'::jsonb,
    citations          jsonb       NOT NULL DEFAULT '[]'::jsonb,
    chat_model         text,
    embed_model        text,
    error              text,
    -- Which rule fired and how each qualifier compared, the public phase
    -- durations, and the exact ready versions searched. Operational metadata
    -- only: no prompt, no model reply, no hidden reasoning, no local path.
    decision_explanation jsonb     NOT NULL DEFAULT '{}'::jsonb,
    timings              jsonb     NOT NULL DEFAULT '{}'::jsonb,
    index_references     jsonb     NOT NULL DEFAULT '[]'::jsonb,
    created_at         timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit_candidate (
    id             bigserial PRIMARY KEY,
    audit_id       bigint  NOT NULL REFERENCES audit_run(id) ON DELETE CASCADE,
    evidence_id    bigint  NOT NULL REFERENCES evidence_unit(id) ON DELETE CASCADE,
    lexical_rank   integer,
    lexical_score  double precision,
    vector_rank    integer,
    vector_score   double precision,
    graph_rank     integer,
    graph_score    double precision,
    combined_rank  integer,
    combined_score double precision NOT NULL DEFAULT 0,
    -- Set when this candidate was pulled in as a neighbour of another one.
    expanded_from  bigint  REFERENCES evidence_unit(id) ON DELETE SET NULL,
    visual_status  text    NOT NULL DEFAULT 'not_applicable'
                   CHECK (visual_status IN ('not_applicable', 'verified',
                                            'rejected', 'unavailable')),
    selected       boolean NOT NULL DEFAULT false,
    reason         text,
    UNIQUE (audit_id, evidence_id)
);

CREATE INDEX IF NOT EXISTS audit_candidate_audit_idx ON audit_candidate (audit_id);


-- In-place upgrade for databases created before these columns existed. The
-- CREATE TABLE statements above are authoritative for a fresh database; these
-- keep `db init` idempotent on an existing one instead of demanding a re-index.
ALTER TABLE document_version ADD COLUMN IF NOT EXISTS attempt integer NOT NULL DEFAULT 1;
ALTER TABLE audit_candidate  ADD COLUMN IF NOT EXISTS lexical_score double precision;
ALTER TABLE audit_candidate  ADD COLUMN IF NOT EXISTS vector_score double precision;
ALTER TABLE audit_candidate  ADD COLUMN IF NOT EXISTS graph_score double precision;
ALTER TABLE audit_candidate  ADD COLUMN IF NOT EXISTS combined_rank integer;
ALTER TABLE audit_candidate  ADD COLUMN IF NOT EXISTS expanded_from bigint;
ALTER TABLE audit_candidate  ADD COLUMN IF NOT EXISTS visual_status text NOT NULL DEFAULT 'not_applicable';
ALTER TABLE audit_candidate  ADD COLUMN IF NOT EXISTS reason text;
ALTER TABLE audit_candidate  DROP COLUMN IF EXISTS note;

DO $$
BEGIN
    ALTER TABLE document_version DROP CONSTRAINT document_version_document_id_fingerprint_key;
EXCEPTION WHEN undefined_object THEN
    NULL;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS document_version_identity_idx
    ON document_version (document_id, fingerprint, attempt);

-- Document identity (schema 3). Deterministic backfill: the stored PDF hash if
-- there is one, else the newest version's output root, else the document id.
-- Document ids and versions are never rewritten, so existing indexes keep
-- serving. The bases match `ingest._identity_key`, which is what stops this
-- migration from splitting a document that is later re-ingested.
ALTER TABLE document ADD COLUMN IF NOT EXISTS identity_key text;

UPDATE document d SET identity_key = encode(sha256(convert_to(
    CASE
        WHEN d.sha256 IS NOT NULL THEN 'pdf:' || d.sha256
        WHEN COALESCE(d.source_uri, '') <> '' THEN 'uri:' || d.source_uri
        ELSE COALESCE(
            (SELECT 'output:' || v.output_root FROM document_version v
              WHERE v.document_id = d.id ORDER BY v.id DESC LIMIT 1),
            'legacy:' || d.id::text)
    END, 'UTF8')), 'hex')
WHERE d.identity_key IS NULL;

ALTER TABLE document ALTER COLUMN identity_key SET NOT NULL;

DO $$
BEGIN
    ALTER TABLE document DROP CONSTRAINT document_name_sha256_key;
EXCEPTION WHEN undefined_object THEN
    NULL;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS document_identity_idx ON document (identity_key);

-- Structured verdict explanation (schema 4). Defaulted rather than backfilled:
-- an audit run before this change genuinely has no explanation, and inventing
-- one would be worse than an empty object a reader can recognize.
ALTER TABLE audit_run ADD COLUMN IF NOT EXISTS
    decision_explanation jsonb NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE audit_run ADD COLUMN IF NOT EXISTS
    timings jsonb NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE audit_run ADD COLUMN IF NOT EXISTS
    index_references jsonb NOT NULL DEFAULT '[]'::jsonb;

INSERT INTO schema_meta (id, version) VALUES (1, 4)
ON CONFLICT (id) DO UPDATE SET version = EXCLUDED.version, applied_at = now();
