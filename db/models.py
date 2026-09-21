"""
SQLAlchemy ORM models — one table per schema section from the dashboard images.
"""
from sqlalchemy import Column, Integer, Float, String, Date, DateTime, Index
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime


class Base(DeclarativeBase):
    pass


class PlatformMetric(Base):
    """Google Real Time dashboard — Platform section (Impr, Clicks, Cost)."""
    __tablename__ = "platform_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False)
    hour = Column(Integer, nullable=False)
    platform = Column(String(20), nullable=False)   # google | meta | microsoft
    impressions = Column(Integer, default=0)
    clicks = Column(Integer, default=0)
    cost = Column(Float, default=0.0)
    conversions = Column(Integer, default=0)
    cpc = Column(Float, default=0.0)
    ctr = Column(Float, default=0.0)
    cvr = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_platform_date_hour", "date", "hour", "platform"),)


class CallCenter(Base):
    """Call Center (Actual) — Calls Revenue, Data Revenue, Total Revenue, Calls, Data-Submit Form."""
    __tablename__ = "call_center"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False)
    hour = Column(Integer, nullable=False)
    calls_revenue = Column(Float, default=0.0)
    data_revenue = Column(Float, default=0.0)
    total_revenue = Column(Float, default=0.0)
    inbound_calls = Column(Integer, default=0)
    data_submit_forms = Column(Integer, default=0)
    medium_income_verified_rate = Column(Float, default=0.0)
    high_income_verified_carrier_rate = Column(Float, default=0.0)
    margin_dollar = Column(Float, default=0.0)
    margin_pct = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_callcenter_date_hour", "date", "hour"),)


class CallQualityControl(Base):
    """Call Quality Control — Medium Income Call Ratio at 30s / 60s / 90s."""
    __tablename__ = "call_quality"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False)
    hour = Column(Integer, nullable=False)
    total_calls = Column(Integer, default=0)
    medium_income_call_ratio_30s = Column(Float, default=0.0)
    medium_income_call_ratio_60s = Column(Float, default=0.0)
    medium_income_call_ratio_90s = Column(Float, default=0.0)
    calls_30s = Column(Integer, default=0)
    calls_60s = Column(Integer, default=0)
    calls_90s = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_callquality_date_hour", "date", "hour"),)


class DataQualityControl(Base):
    """Data Quality Control — High Income Verified Carrier Data Ratio by income tier."""
    __tablename__ = "data_quality"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False)
    hour = Column(Integer, nullable=False)
    high_income_verified_carrier_low_income_ratio = Column(Float, default=0.0)
    high_income_verified_carrier_medium_income_ratio = Column(Float, default=0.0)
    high_income_verified_carrier_high_income_ratio = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_dataquality_date_hour", "date", "hour"),)


class AttributionRate(Base):
    """Attribution Rates — Inbound Call, Medium Income Verified, Any Data, High Income w/ Carrier."""
    __tablename__ = "attribution_rates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False)
    hour = Column(Integer, nullable=False)
    inbound_call_rate = Column(Float, default=0.0)
    medium_income_verified_rate = Column(Float, default=0.0)
    any_data_rate = Column(Float, default=0.0)
    high_income_verified_carrier_rate = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_attribution_date_hour", "date", "hour"),)


class AIInsight(Base):
    """Claude AI generated insights and recommendations."""
    __tablename__ = "ai_insights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    generated_at = Column(DateTime, default=datetime.utcnow)
    insight_type = Column(String(50))   # anomaly | budget_reallocation | fatigue_flag | daily_summary
    platform = Column(String(20), nullable=True)
    summary = Column(String(500))
    full_response = Column(String(5000))
    severity = Column(String(10), default="info")  # info | warning | critical
