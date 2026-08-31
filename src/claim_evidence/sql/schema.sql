-- The one current claim_evidence schema.
--
-- There is exactly one schema asset and no migration history. Development data
-- is disposable and rebuildable: a database that does not match this file is
-- reset and re-indexed, never upgraded in place. That is why there are no
-- ALTER/backfill sections here -- an in-place upgrade path is deferred until a
-- schema is declared stable (DG-01), and pretending to have one is how a
-- half-migrated database ends up serving confident wrong citations.
--
-- Re-runnable against a database this same file created: every object is
-- CREATE ... IF NOT EXISTS, so a repeated `db init` is a no-op. `EMBED_DIM` is
-- substituted with the configured embedding dimension before execution.

CREATE EXTENSION IF NOT EXISTS vector;

-- What installed this database, and when. `db init` compares the recorded
-- version and schema-file digest against the package's own before it will
-- treat an existing database as current.
CREATE TABLE IF NOT EXISTS schema_meta (
    id                integer     PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    version           integer     NOT NULL,
    schema_sql_sha256 text        NOT NULL,
    initialized_at    timestamptz NOT NULL DEFAULT now(),
    applied_at        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document (
    id           bigserial PRIMARY KEY,
    name         text        NOT NULL,
    sha256       text,
    source_uri   text,
    -- Internal identity only, never a public API field: sha256 of a
    -- namespace-tagged basis -- the PDF hash, else the logical source URI, else
    -- the canonical output root -- so equal text in different namespaces cannot
    -- collide, and two output roots sharing a basename stay two documents.
    identity_key text        NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS document_identity_idx ON document (identity_key);

-- A version is queryable only in a queryable status. An interrupted build is
-- reconciled to 'interrupted' on restart and is simply never joined against.
CREATE TABLE IF NOT EXISTS document_version (
    id            bigserial PRIMARY KEY,
    document_id   bigint      NOT NULL REFERENCES document(id) ON DELETE CASCADE,
    fingerprint   text        NOT NULL,
    embed_model   text        NOT NULL,
    embed_dim     integer     NOT NULL,
    status        text        NOT NULL DEFAULT 'building'
                  CHECK (status IN ('building', 'ready', 'degraded', 'failed',
                                    'inactive', 'interrupted')),
    output_root   text        NOT NULL,
    source_pdf    text,
    -- A forced rebuild re-indexes an unchanged source, so identity is
    -- (document, fingerprint, attempt): the new 'building' row coexists with
    -- the old ready one, which keeps serving queries until the swap.
    attempt       integer     NOT NULL DEFAULT 1,
    created_at    timestamptz NOT NULL DEFAULT now(),
    ready_at      timestamptz,
    failed_at     timestamptz,
    failure_code  text,
    failure_phase text,
    last_progress_at timestamptz NOT NULL DEFAULT now(),
    -- Narrative fact enrichment is optional; these say how much of it landed,
    -- so 'degraded' is a measured state rather than a label.
    fact_candidates_total     integer NOT NULL DEFAULT 0,
    fact_candidates_succeeded integer NOT NULL DEFAULT 0,
    UNIQUE (document_id, fingerprint, attempt)
);

CREATE INDEX IF NOT EXISTS document_version_ready_idx
    ON document_version (document_id, status);
CREATE INDEX IF NOT EXISTS document_version_progress_idx
    ON document_version (status, last_progress_at);

-- One failed narrative-fact candidate. The evidence key and a safe reason code
-- only: the passage itself and the model's error text stay out of the database
-- so a retry can be targeted without storing what went wrong verbatim.
CREATE TABLE IF NOT EXISTS fact_candidate_failure (
    version_id  bigint      NOT NULL REFERENCES document_version(id) ON DELETE CASCADE,
    unit_key    text        NOT NULL,
    reason_code text        NOT NULL,
    recorded_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (version_id, unit_key)
);

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
    -- Root-relative, contained, and checked against the extraction root before
    -- the version can activate.
    artifact_path      text    NOT NULL,
    geometry_precision text    NOT NULL,
    truncated_source   boolean NOT NULL DEFAULT false,
    -- Where this unit sits on its page and what it belongs to. Context
    -- expansion reads these instead of evidence ids, which record when a row
    -- was inserted rather than what the page actually says.
    source_order       integer NOT NULL,
    context_key        text,
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
CREATE INDEX IF NOT EXISTS evidence_unit_source_order_idx
    ON evidence_unit (page_id, source_order);
CREATE INDEX IF NOT EXISTS evidence_unit_context_idx
    ON evidence_unit (page_id, context_key);

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
    id               bigserial PRIMARY KEY,
    entity_id        bigint NOT NULL REFERENCES entity(id) ON DELETE CASCADE,
    alias            text   NOT NULL,
    normalized_alias text   NOT NULL,
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
    status             text        NOT NULL DEFAULT 'running'
                       CHECK (status IN ('running', 'completed', 'failed', 'interrupted')),
    -- The corpus this audit searched, recorded when it opened. Not derived from
    -- citations: an insufficient verdict cites nothing and still searched
    -- something.
    requested_document_ids jsonb   NOT NULL DEFAULT '[]'::jsonb,
    verdict            text,
    rationale          text,
    evidence_quality   text,
    missing_qualifiers jsonb       NOT NULL DEFAULT '[]'::jsonb,
    citations          jsonb       NOT NULL DEFAULT '[]'::jsonb,
    chat_model         text,
    embed_model        text,
    error              text,
    -- Which rule fired and how each qualifier compared, the public phase
    -- durations, and the exact versions searched. Operational metadata only:
    -- no prompt, no model reply, no hidden reasoning, no local path.
    decision_explanation jsonb     NOT NULL DEFAULT '{}'::jsonb,
    timings              jsonb     NOT NULL DEFAULT '{}'::jsonb,
    index_references     jsonb     NOT NULL DEFAULT '[]'::jsonb,
    created_at         timestamptz NOT NULL DEFAULT now(),
    completed_at       timestamptz,
    failed_at          timestamptz,
    failure_code       text,
    failure_phase      text,
    retryable          boolean
);

CREATE INDEX IF NOT EXISTS audit_run_status_idx ON audit_run (status, created_at);

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

-- One re-verified visual crop. The visible text the model reported and a safe
-- result/reason code -- never its reasoning, and never the prompt.
CREATE TABLE IF NOT EXISTS visual_verification (
    id           bigserial PRIMARY KEY,
    audit_id     bigint  NOT NULL REFERENCES audit_run(id) ON DELETE CASCADE,
    evidence_id  bigint  NOT NULL REFERENCES evidence_unit(id) ON DELETE CASCADE,
    result       text    NOT NULL
                 CHECK (result IN ('support', 'conflict', 'illegible', 'unrelated')),
    reason_code  text    NOT NULL,
    visible_text text    NOT NULL DEFAULT '',
    checked_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (audit_id, evidence_id)
);
