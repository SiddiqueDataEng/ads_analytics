"""
Entry point — FastAPI app with lifespan startup (DB init + seed + scheduler).
Run with: uvicorn main:app --reload --port 8000
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from db.database import init_db, AsyncSessionLocal
from data_generators.ads_generator import generate_historical, generate_full_day
from pipeline.ingest import ingest_all
from scheduler import start_scheduler
from api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init DB tables
    await init_db()

    # Seed 30 days of historical data on first run
    async with AsyncSessionLocal() as session:
        from sqlalchemy import select, func
        from db.models import PlatformMetric
        count = (await session.execute(select(func.count()).select_from(PlatformMetric))).scalar()
        if count == 0:
            print("Seeding 30 days of historical data...")
            historical = generate_historical(30)
            await ingest_all(session, historical)
            today = generate_full_day()
            await ingest_all(session, today)
            print("Seed complete.")

    # Start background scheduler
    start_scheduler()
    yield


app = FastAPI(title="Ads Performance Dashboard", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

# Serve frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
