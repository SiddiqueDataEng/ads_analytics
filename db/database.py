"""Async SQLAlchemy engine + session + DB init."""
import os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from db.models import Base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./ads_dashboard.db")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Columns added after the original schema shipped. create_all() only creates
# missing TABLES, not missing COLUMNS on existing tables — SQLite needs an
# explicit ALTER TABLE for that, so we do it here rather than requiring
# everyone to delete ads_dashboard.db and lose their history.
_NEW_COLUMNS = {
    "platform_metrics": [
        ("campaign_id", "VARCHAR(64)"),
        ("campaign_name", "VARCHAR(255)"),
        ("source", "VARCHAR(10) DEFAULT 'synthetic'"),
    ],
}


async def _migrate_existing_columns(conn):
    for table, columns in _NEW_COLUMNS.items():
        result = await conn.execute(text(f"PRAGMA table_info({table})"))
        existing = {row[1] for row in result.fetchall()}
        for col_name, col_def in columns:
            if col_name not in existing:
                await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}"))


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # only meaningful for sqlite; other DBs would use Alembic in production
        if DATABASE_URL.startswith("sqlite"):
            await _migrate_existing_columns(conn)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
