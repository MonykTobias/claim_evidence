-- claim_evidence schema.
--
-- Re-runnable: every object is created IF NOT EXISTS so `db init` is safe on a
-- populated database. `EMBED_DIM` is substituted with the configured embedding
-- dimension before execution.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document (
    id           bigserial PRIMARY KEY,
    name         text        NOT NULL,
    sha256       text,
    source_uri   text,
    created_at   timestamptz NOT NULL DEFAULT now(),
    -- NULLS NOT DISTINCT matters: ingesting without a source PDF leaves sha256
    -- NULL, and the default NULL-is-distinct rule would make every re-ingest a
    -- brand new document instead of matching the existing one.
    UNIQUE NULLS NOT DISTINCT (name, sha256)
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
    created_at    timestamptz NOT NULL DEFAULT now(),
    ready_at      timestamptz,
    UNIQUE (document_id, fingerprint)
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
    created_at         timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit_candidate (
    id             bigserial PRIMARY KEY,
    audit_id       bigint  NOT NULL REFERENCES audit_run(id) ON DELETE CASCADE,
    evidence_id    bigint  NOT NULL REFERENCES evidence_unit(id) ON DELETE CASCADE,
    lexical_rank   integer,
    vector_rank    integer,
    graph_rank     integer,
    combined_score double precision NOT NULL DEFAULT 0,
    selected       boolean NOT NULL DEFAULT false,
    note           text,
    UNIQUE (audit_id, evidence_id)
);

CREATE INDEX IF NOT EXISTS audit_candidate_audit_idx ON audit_candidate (audit_id);
