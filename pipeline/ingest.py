"""
Data ingestion pipeline — loads generated (or real API) data into the DB.
Handles deduplication and upsert logic per date+hour+platform.
"""
from typing import Dict, List, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from db.models import (
    PlatformMetric, CallCenter, CallQualityControl,
    DataQualityControl, AttributionRate
)


async def ingest_platform_metrics(session: AsyncSession, rows: List[Dict[str, Any]]):
    for row in rows:
        # delete existing row for this date/hour/platform to avoid dupes
        await session.execute(
            delete(PlatformMetric).where(
                PlatformMetric.date == row["date"],
                PlatformMetric.hour == row["hour"],
                PlatformMetric.platform == row["platform"],
            )
        )
        session.add(PlatformMetric(**row))
    await session.commit()


async def ingest_call_center(session: AsyncSession, rows: List[Dict[str, Any]]):
    for row in rows:
        await session.execute(
            delete(CallCenter).where(
                CallCenter.date == row["date"],
                CallCenter.hour == row["hour"],
            )
        )
        session.add(CallCenter(**row))
    await session.commit()


async def ingest_call_quality(session: AsyncSession, rows: List[Dict[str, Any]]):
    for row in rows:
        await session.execute(
            delete(CallQualityControl).where(
                CallQualityControl.date == row["date"],
                CallQualityControl.hour == row["hour"],
            )
        )
        session.add(CallQualityControl(**row))
    await session.commit()


async def ingest_data_quality(session: AsyncSession, rows: List[Dict[str, Any]]):
    for row in rows:
        await session.execute(
            delete(DataQualityControl).where(
                DataQualityControl.date == row["date"],
                DataQualityControl.hour == row["hour"],
            )
        )
        session.add(DataQualityControl(**row))
    await session.commit()


async def ingest_attribution_rates(session: AsyncSession, rows: List[Dict[str, Any]]):
    for row in rows:
        await session.execute(
            delete(AttributionRate).where(
                AttributionRate.date == row["date"],
                AttributionRate.hour == row["hour"],
            )
        )
        session.add(AttributionRate(**row))
    await session.commit()


async def ingest_all(session: AsyncSession, data: Dict[str, List[Dict]]):
    """Ingest a full batch from the generator into all tables."""
    await ingest_platform_metrics(session, data.get("platform_metrics", []))
    await ingest_call_center(session, data.get("call_center", []))
    await ingest_call_quality(session, data.get("call_quality", []))
    await ingest_data_quality(session, data.get("data_quality", []))
    await ingest_attribution_rates(session, data.get("attribution_rates", []))
