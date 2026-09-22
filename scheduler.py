"""
APScheduler jobs — runs hourly data ingestion and daily AI analysis.

DATA_SOURCE env var controls where ingestion pulls from:
    DATA_SOURCE=live       -> Google/Meta/Microsoft APIs via connectors/live.py
    DATA_SOURCE=synthetic  -> data_generators/ads_generator.py (default, no credentials needed)

In "live" mode, if a platform's connector fails (bad token, API outage), we
still ingest whatever platforms succeeded and log the failure — we do NOT
silently fall back to synthetic numbers for a live account, since that would
quietly corrupt real reporting with fake data. If ALL platforms fail, we
log loudly; the dashboard will just show stale data for that hour rather
than fabricated numbers.
"""
import os
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime

from db.database import AsyncSessionLocal
from data_generators.ads_generator import generate_current_hour, generate_full_day
from pipeline.ingest import ingest_all
from ai.claude_engine import ClaudeEngine

claude = ClaudeEngine()
scheduler = AsyncIOScheduler()

DATA_SOURCE = os.getenv("DATA_SOURCE", "synthetic").lower()


async def hourly_ingest():
    """Ingest current-hour data into DB — live API data or synthetic, per DATA_SOURCE."""
    now = datetime.utcnow()
    date, hour = now.strftime("%Y-%m-%d"), now.hour

    if DATA_SOURCE == "live":
        from connectors.live import fetch_all_hourly
        report = fetch_all_hourly(date, hour)
        if report.errors:
            for platform, err in report.errors.items():
                print(f"[{now.isoformat()}] LIVE INGEST FAILED — {platform}: {err}")
        if not report.payload["platform_metrics"]:
            print(f"[{now.isoformat()}] Hourly ingest skipped — no platforms returned data.")
            return
        async with AsyncSessionLocal() as session:
            await ingest_all(session, report.payload)
        print(f"[{now.isoformat()}] Hourly ingest complete "
              f"(live: {report.succeeded}, failed: {list(report.errors)}).")
    else:
        async with AsyncSessionLocal() as session:
            data = generate_current_hour()
            await ingest_all(session, data)
        print(f"[{now.isoformat()}] Hourly ingest complete (synthetic).")


async def daily_ai_analysis():
    """Run Claude anomaly detection + daily brief once per day."""
    date = datetime.utcnow().strftime("%Y-%m-%d")
    async with AsyncSessionLocal() as session:
        await claude.run_anomaly_detection(session, date)
        await claude.run_daily_brief(session, date)
        await claude.run_budget_reallocation(session, date)
        await claude.run_fatigue_scan(session, date)
    print(f"[{datetime.utcnow().isoformat()}] Daily AI analysis complete.")


async def weekly_ai_strategy():
    """Run Claude's weekly cross-channel strategy recommendation."""
    date = datetime.utcnow().strftime("%Y-%m-%d")
    async with AsyncSessionLocal() as session:
        await claude.run_weekly_strategy(session, date)
    print(f"[{datetime.utcnow().isoformat()}] Weekly AI strategy complete.")


def start_scheduler():
    scheduler.add_job(hourly_ingest, "cron", minute=5)                  # runs at :05 every hour
    scheduler.add_job(daily_ai_analysis, "cron", hour=7, minute=0)      # 7am UTC daily
    scheduler.add_job(weekly_ai_strategy, "cron", day_of_week="mon", hour=7, minute=30)  # Monday 7:30am UTC
    scheduler.start()
    print(f"Scheduler started — DATA_SOURCE={DATA_SOURCE}")
