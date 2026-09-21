"""FastAPI routes — dashboard data, KPIs, AI insights."""
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from db.database import get_db
from db.models import AIInsight
from pipeline.transforms import (
    get_topline_kpis, get_realtime_hourly, get_wow_trend,
    get_mtd_summary, get_call_quality_summary,
    get_data_quality_summary, get_attribution_summary,
)
from ai.claude_engine import ClaudeEngine

router = APIRouter()
claude = ClaudeEngine()


def today_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


@router.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@router.get("/dashboard/realtime")
async def realtime(date: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    return await get_realtime_hourly(db, date or today_str())


@router.get("/dashboard/kpis")
async def kpis(date: Optional[str] = None, platform: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    return await get_topline_kpis(db, date or today_str(), platform)


@router.get("/dashboard/wow")
async def wow(date: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    return await get_wow_trend(db, date or today_str())


@router.get("/dashboard/mtd")
async def mtd(date: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    return await get_mtd_summary(db, date or today_str())


@router.get("/dashboard/call-quality")
async def call_quality(date: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    return await get_call_quality_summary(db, date or today_str())


@router.get("/dashboard/data-quality")
async def data_quality(date: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    return await get_data_quality_summary(db, date or today_str())


@router.get("/dashboard/attribution")
async def attribution(date: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    return await get_attribution_summary(db, date or today_str())


@router.post("/ai/anomaly-detection")
async def run_anomaly(date: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    result = await claude.run_anomaly_detection(db, date or today_str())
    return {"id": result.id, "severity": result.severity, "summary": result.summary, "full": result.full_response}


@router.post("/ai/daily-brief")
async def run_daily_brief(date: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    result = await claude.run_daily_brief(db, date or today_str())
    return {"id": result.id, "brief": result.full_response}


@router.get("/ai/insights")
async def get_insights(limit: int = 10, db: AsyncSession = Depends(get_db)):
    q = select(AIInsight).order_by(desc(AIInsight.generated_at)).limit(limit)
    rows = (await db.execute(q)).scalars().all()
    return [
        {
            "id": r.id,
            "type": r.insight_type,
            "platform": r.platform,
            "severity": r.severity,
            "summary": r.summary,
            "generated_at": r.generated_at.isoformat() if r.generated_at else None,
        }
        for r in rows
    ]
