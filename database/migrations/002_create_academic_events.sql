
-- =====================================================
-- Migration: Create Academic Events
-- Project: Q-TIME
-- =====================================================

CREATE TABLE IF NOT EXISTS academic_events (

    id BIGSERIAL PRIMARY KEY,

    document_id BIGINT NOT NULL
        REFERENCES documents(id)
        ON DELETE RESTRICT,

    source_key VARCHAR(150) NOT NULL,

    academic_year VARCHAR(20) NOT NULL,

    semester VARCHAR(20) NOT NULL
        CHECK (
            semester IN ('Ganjil', 'Genap', 'Umum')
        ),

    event_name TEXT NOT NULL,

    event_category VARCHAR(50) NOT NULL,

    start_date DATE NOT NULL,

    end_date DATE NOT NULL,

    status VARCHAR(20) NOT NULL DEFAULT 'needs_review'
        CHECK (
            status IN (
                'draft',
                'needs_review',
                'published',
                'archived'
            )
        ),

    source_page INTEGER NOT NULL
        CHECK (source_page > 0),

    source_text TEXT NOT NULL,

    review_notes TEXT,

    verified_by VARCHAR(150),

    verified_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_event_dates
        CHECK (end_date >= start_date),

    CONSTRAINT valid_published_event
        CHECK (
            status <> 'published'
            OR (
                verified_by IS NOT NULL
                AND verified_at IS NOT NULL
            )
        ),

    CONSTRAINT unique_document_event
        UNIQUE (document_id, source_key)
);


-- Index for academic event filtering
CREATE INDEX IF NOT EXISTS idx_academic_events_lookup
ON academic_events (
    academic_year,
    semester,
    status,
    start_date
);


-- Index for document reference
CREATE INDEX IF NOT EXISTS idx_academic_events_document
ON academic_events(document_id);


-- Index for category
CREATE INDEX IF NOT EXISTS idx_academic_events_category
ON academic_events(event_category);
