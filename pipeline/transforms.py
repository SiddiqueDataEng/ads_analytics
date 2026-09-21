"""
Data transformation layer — computes KPIs, pacing, WoW, MTD from raw tables.
Matches the Topline Table KPIs from the dashboard: CPL, RPL, CPC, CVR, Margin $, Margin %
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from db.models import PlatformMetric, CallCenter, CallQualityControl, DataQualityControl, AttributionRate


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    return round(numerator / denominator, 4) if denominator else default


async def get_topline_kpis(session: AsyncSession, date: str, platform: Optional[str] = None) -> Dict:
    """
    Compute Topline Table KPIs for a given date:
    CPL, RPL, CPC, CVR, Margin $, Margin %
    """
    q = select(
        func.sum(PlatformMetric.cost).label("total_cost"),
        func.sum(PlatformMetric.clicks).label("total_clicks"),
        func.sum(PlatformMetric.impressions).label("total_impressions"),
        func.sum(PlatformMetric.conversions).label("total_conversions"),
    ).where(PlatformMetric.date == date)
    if platform:
        q = q.where(PlatformMetric.platform == platform)
    pm = (await session.execute(q)).one()

    cc_q = select(
        func.sum(CallCenter.total_revenue).label("total_revenue"),
        func.sum(CallCenter.inbound_calls).label("total_calls"),
    ).where(CallCenter.date == date)
    cc = (await session.execute(cc_q)).one()

    total_cost = pm.total_cost or 0.0
    total_clicks = pm.total_clicks or 0
    total_conversions = pm.total_conversions or 0
    total_revenue = cc.total_revenue or 0.0
    total_leads = total_conversions

    cpl = safe_divide(total_cost, total_leads)          # Cost per Lead
    rpl = safe_divide(total_revenue, total_leads)       # Revenue per Lead
    cpc = safe_divide(total_cost, total_clicks)         # Cost per Click
    cvr = safe_divide(total_conversions, total_clicks)  # Conversion Rate
    margin_dollar = round(total_revenue - total_cost, 2)
    margin_pct = safe_divide(margin_dollar, total_revenue)

    return {
        "date": date,
        "platform": platform or "all",
        "total_cost": round(total_cost, 2),
        "total_clicks": total_clicks,
        "total_impressions": pm.total_impressions or 0,
        "total_conversions": total_conversions,
        "total_revenue": round(total_revenue, 2),
        "CPL": cpl,
        "RPL": rpl,
        "CPC": cpc,
        "CVR": cvr,
        "margin_dollar": margin_dollar,
        "margin_pct": margin_pct,
    }


async def get_realtime_hourly(session: AsyncSession, date: str) -> List[Dict]:
    """Google Real-Time dashboard — hourly breakdown by platform."""
    q = select(
        PlatformMetric.hour,
        PlatformMetric.platform,
        PlatformMetric.impressions,
        PlatformMetric.clicks,
        PlatformMetric.cost,
        CallCenter.calls_revenue,
        CallCenter.data_revenue,
        CallCenter.total_revenue,
        CallCenter.inbound_calls,
        CallCenter.data_submit_forms,
    ).outerjoin(
        CallCenter,
        and_(CallCenter.date == PlatformMetric.date, CallCenter.hour == PlatformMetric.hour)
    ).where(PlatformMetric.date == date).order_by(PlatformMetric.hour, PlatformMetric.platform)

    rows = (await session.execute(q)).all()
    return [dict(r._mapping) for r in rows]


async def get_wow_trend(session: AsyncSession, date: str) -> Dict:
    """Week-over-Week comparison for the given date vs same day last week."""
    last_week = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
    current = await get_topline_kpis(session, date)
    previous = await get_topline_kpis(session, last_week)

    def pct_change(curr, prev):
        return safe_divide(curr - prev, abs(prev)) if prev else 0.0

    return {
        "current_week": current,
        "previous_week": previous,
        "wow_cost_pct": pct_change(current["total_cost"], previous["total_cost"]),
        "wow_revenue_pct": pct_change(current["total_revenue"], previous["total_revenue"]),
        "wow_margin_pct": pct_change(current["margin_dollar"], previous["margin_dollar"]),
        "wow_cvr_pct": pct_change(current["CVR"], previous["CVR"]),
    }


async def get_mtd_summary(session: AsyncSession, date: str) -> Dict:
    """Month-to-Date aggregate from first of month to given date."""
    month_start = date[:8] + "01"
    q = select(
        func.sum(PlatformMetric.cost).label("mtd_cost"),
        func.sum(PlatformMetric.clicks).label("mtd_clicks"),
        func.sum(PlatformMetric.conversions).label("mtd_conversions"),
    ).where(
        and_(PlatformMetric.date >= month_start, PlatformMetric.date <= date)
    )
    pm = (await session.execute(q)).one()

    cc_q = select(
        func.sum(CallCenter.total_revenue).label("mtd_revenue"),
        func.sum(CallCenter.margin_dollar).label("mtd_margin"),
    ).where(
        and_(CallCenter.date >= month_start, CallCenter.date <= date)
    )
    cc = (await session.execute(cc_q)).one()

    mtd_cost = pm.mtd_cost or 0.0
    mtd_revenue = cc.mtd_revenue or 0.0
    mtd_margin = cc.mtd_margin or 0.0

    return {
        "month_start": month_start,
        "through_date": date,
        "mtd_cost": round(mtd_cost, 2),
        "mtd_clicks": pm.mtd_clicks or 0,
        "mtd_conversions": pm.mtd_conversions or 0,
        "mtd_revenue": round(mtd_revenue, 2),
        "mtd_margin_dollar": round(mtd_margin, 2),
        "mtd_margin_pct": safe_divide(mtd_margin, mtd_revenue),
        "mtd_cpl": safe_divide(mtd_cost, pm.mtd_conversions or 0),
        "mtd_rpl": safe_divide(mtd_revenue, pm.mtd_conversions or 0),
    }


async def get_call_quality_summary(session: AsyncSession, date: str) -> List[Dict]:
    q = select(CallQualityControl).where(CallQualityControl.date == date).order_by(CallQualityControl.hour)
    rows = (await session.execute(q)).scalars().all()
    return [
        {
            "hour": r.hour,
            "total_calls": r.total_calls,
            "medium_income_call_ratio_30s": r.medium_income_call_ratio_30s,
            "medium_income_call_ratio_60s": r.medium_income_call_ratio_60s,
            "medium_income_call_ratio_90s": r.medium_income_call_ratio_90s,
        }
        for r in rows
    ]


async def get_data_quality_summary(session: AsyncSession, date: str) -> List[Dict]:
    q = select(DataQualityControl).where(DataQualityControl.date == date).order_by(DataQualityControl.hour)
    rows = (await session.execute(q)).scalars().all()
    return [
        {
            "hour": r.hour,
            "low_income_ratio": r.high_income_verified_carrier_low_income_ratio,
            "medium_income_ratio": r.high_income_verified_carrier_medium_income_ratio,
            "high_income_ratio": r.high_income_verified_carrier_high_income_ratio,
        }
        for r in rows
    ]


async def get_attribution_summary(session: AsyncSession, date: str) -> List[Dict]:
    q = select(AttributionRate).where(AttributionRate.date == date).order_by(AttributionRate.hour)
    rows = (await session.execute(q)).scalars().all()
    return [
        {
            "hour": r.hour,
            "inbound_call_rate": r.inbound_call_rate,
            "medium_income_verified_rate": r.medium_income_verified_rate,
            "any_data_rate": r.any_data_rate,
            "high_income_verified_carrier_rate": r.high_income_verified_carrier_rate,
        }
        for r in rows
    ]
