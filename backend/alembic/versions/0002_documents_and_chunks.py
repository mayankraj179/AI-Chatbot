"""documents and chunks

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("assistant_id", sa.String(64), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("media_type", sa.String(128), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("embedding_model", sa.String(200), nullable=False),
        sa.Column("access_labels", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        # Set when the source file disappears; the row and its chunks are kept.
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("tenant_id", "assistant_id", "source_uri"),
        sa.CheckConstraint("cardinality(access_labels) > 0", name="documents_labels_nonempty"),
    )
    op.create_table(
        "chunks",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("assistant_id", sa.String(64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("heading_path", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("page", sa.Integer(), nullable=True),
        # Copied from the document so the search WHERE clause filters without a join.
        sa.Column("access_labels", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("embedding_model", sa.String(200), nullable=False),
        # Older versions are flagged, never deleted; search only reads active chunks.
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("document_id", "version", "chunk_index"),
        sa.CheckConstraint("cardinality(access_labels) > 0", name="chunks_labels_nonempty"),
    )
    # Dimensionless so models with different sizes can coexist; rows are only ever
    # compared within one embedding_model.
    op.execute("ALTER TABLE chunks ADD COLUMN embedding vector NOT NULL")
    op.create_index(
        "ix_chunks_active_scope",
        "chunks",
        ["tenant_id", "assistant_id", "embedding_model"],
        postgresql_where=sa.text("retired_at IS NULL"),
    )
    op.create_index("ix_chunks_access_labels", "chunks", ["access_labels"], postgresql_using="gin")


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("documents")
