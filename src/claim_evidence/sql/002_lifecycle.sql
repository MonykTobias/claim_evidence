-- Build lifecycle, audit lifecycle, and source-relationship columns.
--
-- Sequential and idempotent, like 001: `db init` runs every file in this
-- directory in name order on every invocation, so this one must be safe to
-- re-run against a database that already has it.
--
-- Backfills never change an existing row's status, and never invent a failure
-- cause for work whose outcome was not recorded.

-- --- M-2: an interrupted or failed build must be distinguishable -----------

ALTER TABLE document_version ADD COLUMN IF NOT EXISTS failed_at     timestamptz;
ALTER TABLE document_version ADD COLUMN IF NOT EXISTS failure_code  text;
ALTER TABLE document_version ADD COLUMN IF NOT EXISTS failure_phase text;
ALTER TABLE document_version ADD COLUMN IF NOT EXISTS last_progress_at timestamptz;

-- A version that predates this column has no progress history; its ready or
-- creation time is the last moment we can honestly claim it moved.
UPDATE document_version
   SET last_progress_at = COALESCE(ready_at, created_at)
 WHERE last_progress_at IS NULL;

ALTER TABLE document_version ALTER COLUMN last_progress_at SET NOT NULL;
ALTER TABLE document_version ALTER COLUMN last_progress_at SET DEFAULT now();

-- 'failed' joins the existing statuses. Dropping and recreating the check is
-- the only way to widen it, and both halves are guarded so a re-run is a no-op.
DO $$
BEGIN
    ALTER TABLE document_version DROP CONSTRAINT document_version_status_check;
EXCEPTION WHEN undefined_object THEN
    NULL;
END $$;

ALTER TABLE document_version ADD CONSTRAINT document_version_status_check
    CHECK (status IN ('building', 'ready', 'inactive', 'failed'));

-- Health reads recently-active builds and stale ones separately.
CREATE INDEX IF NOT EXISTS document_version_progress_idx
    ON document_version (status, last_progress_at);

-- --- M-4: an audit's corpus and outcome, independent of its citations ------

ALTER TABLE audit_run ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'running';
ALTER TABLE audit_run ADD COLUMN IF NOT EXISTS requested_document_ids jsonb
    NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE audit_run ADD COLUMN IF NOT EXISTS completed_at  timestamptz;
ALTER TABLE audit_run ADD COLUMN IF NOT EXISTS failed_at     timestamptz;
ALTER TABLE audit_run ADD COLUMN IF NOT EXISTS failure_code  text;
ALTER TABLE audit_run ADD COLUMN IF NOT EXISTS failure_phase text;
ALTER TABLE audit_run ADD COLUMN IF NOT EXISTS retryable     boolean;

-- An audit that reached a verdict finished, whatever the verdict was. One that
-- did not stays 'running': the read layer may call an old row stale, but the
-- database must not manufacture a failure that was never observed.
UPDATE audit_run
   SET status = 'completed', completed_at = COALESCE(completed_at, created_at)
 WHERE verdict IS NOT NULL AND status = 'running';

DO $$
BEGIN
    ALTER TABLE audit_run DROP CONSTRAINT audit_run_status_check;
EXCEPTION WHEN undefined_object THEN
    NULL;
END $$;

ALTER TABLE audit_run ADD CONSTRAINT audit_run_status_check
    CHECK (status IN ('running', 'completed', 'failed'));

-- --- M-9: context expansion follows the source, not insertion order --------

-- Deliberately nullable with no backfill. A value derived from evidence ids
-- would be the very thing this replaces, dressed up as provenance; a version
-- indexed before this column simply has no source order, and `neighbours()`
-- falls back to page order for it until the document is re-indexed.
ALTER TABLE evidence_unit ADD COLUMN IF NOT EXISTS source_order integer;
ALTER TABLE evidence_unit ADD COLUMN IF NOT EXISTS context_key  text;

CREATE INDEX IF NOT EXISTS evidence_unit_source_order_idx
    ON evidence_unit (page_id, source_order);
CREATE INDEX IF NOT EXISTS evidence_unit_context_idx
    ON evidence_unit (page_id, context_key);

INSERT INTO schema_meta (id, version) VALUES (1, 5)
ON CONFLICT (id) DO UPDATE SET version = EXCLUDED.version, applied_at = now();
