"""
Reconciliation — compares what's stored in the DB for a given date against
a fresh pull straight from each platform's API, so discrepancies (missed
ingests, API attribution lag, currency/timezone mismatches) surface before
a client notices their native Google Ads UI doesn't match the dashboard.

Only meaningful with DATA_SOURCE=live and real credentials — this makes a
live API call to re-fetch the day, so don't run it on every dashboard
pageview; use it for periodic spot-checks or before a handover call.
"""
from typing import Dict
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from db.models import PlatformMetric

TOLERANCE_PCT = 0.02  # >2% difference gets flagged; ad platforms often have minor same-day attribution drift


async def reconcile_day(session: AsyncSession, date: str) -> Dict:
    from connectors.live import fetch_all_full_day  # imported lazily so synthetic-mode users don't need the SDKs installed

    live_report = fetch_all_full_day(date)
    live_totals: Dict[str, Dict[str, float]] = {}
    for row in live_report.payload["platform_metrics"]:
        p = row["platform"]
        bucket = live_totals.setdefault(p, {"cost": 0.0, "clicks": 0, "impressions": 0, "conversions": 0.0})
        bucket["cost"] += row["cost"]
        bucket["clicks"] += row["clicks"]
        bucket["impressions"] += row["impressions"]
        bucket["conversions"] += row["conversions"]

    q = select(
        PlatformMetric.platform,
        func.sum(PlatformMetric.cost).label("cost"),
        func.sum(PlatformMetric.clicks).label("clicks"),
        func.sum(PlatformMetric.impressions).label("impressions"),
        func.sum(PlatformMetric.conversions).label("conversions"),
    ).where(PlatformMetric.date == date).group_by(PlatformMetric.platform)
    db_rows = (await session.execute(q)).all()
    db_totals = {r.platform: {"cost": r.cost or 0.0, "clicks": r.clicks or 0,
                               "impressions": r.impressions or 0, "conversions": r.conversions or 0.0}
                 for r in db_rows}

    results = []
    for platform in set(list(live_totals) + list(db_totals)):
        live_v = live_totals.get(platform, {"cost": 0.0, "clicks": 0, "impressions": 0, "conversions": 0.0})
        db_v = db_totals.get(platform, {"cost": 0.0, "clicks": 0, "impressions": 0, "conversions": 0.0})
        row = {"platform": platform, "live": live_v, "stored": db_v, "diffs": {}}
        for metric in ["cost", "clicks", "impressions", "conversions"]:
            live_val, db_val = live_v[metric], db_v[metric]
            diff_pct = abs(live_val - db_val) / live_val if live_val else (1.0 if db_val else 0.0)
            row["diffs"][metric] = {
                "diff_pct": round(diff_pct * 100, 2),
                "flagged": diff_pct > TOLERANCE_PCT,
            }
        row["status"] = "MISMATCH" if any(d["flagged"] for d in row["diffs"].values()) else "OK"
        results.append(row)

    return {
        "date": date,
        "connector_errors": live_report.errors,
        "platforms": results,
        "overall_status": "MISMATCH" if any(r["status"] == "MISMATCH" for r in results) else "OK",
    }
