"""
APScheduler jobs — runs hourly data ingestion and daily AI analysis.
"""
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime

from db.database import AsyncSessionLocal
from data_generators.ads_generator import generate_current_hour, generate_full_day
from pipeline.ingest import ingest_all
from ai.claude_engine import ClaudeEngine

claude = ClaudeEngine()
scheduler = AsyncIOScheduler()


async def hourly_ingest():
    """Ingest current-hour mock/real data into DB."""
    async with AsyncSessionLocal() as session:
        data = generate_current_hour()
        await ingest_all(session, data)
    print(f"[{datetime.utcnow().isoformat()}] Hourly ingest complete.")


async def daily_ai_analysis():
    """Run Claude anomaly detection + daily brief once per day."""
    date = datetime.utcnow().strftime("%Y-%m-%d")
    async with AsyncSessionLocal() as session:
        await claude.run_anomaly_detection(session, date)
        await claude.run_daily_brief(session, date)
    print(f"[{datetime.utcnow().isoformat()}] Daily AI analysis complete.")


def start_scheduler():
    scheduler.add_job(hourly_ingest, "cron", minute=5)           # runs at :05 every hour
    scheduler.add_job(daily_ai_analysis, "cron", hour=7, minute=0)  # 7am UTC daily
    scheduler.start()
