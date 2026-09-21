"""
Realistic ads data generator for Google, Meta, and Microsoft Ads.
Generates intraday/hourly data matching the dashboard schema from the project specs.
"""
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any


PLATFORMS = ["google", "meta", "microsoft"]

CAMPAIGN_TYPES = {
    "google": ["Search - Brand", "Search - Non-Brand", "Performance Max", "Display"],
    "meta": ["Prospecting - Interest", "Retargeting - Website", "Lookalike - LTV", "Lead Gen"],
    "microsoft": ["Search - Brand", "Search - Non-Brand", "Audience Network"],
}

INCOME_SEGMENTS = ["low_income", "medium_income", "high_income"]

# Realistic hourly traffic multipliers (0=midnight, peak at 10am-2pm)
HOURLY_MULTIPLIERS = [
    0.1, 0.05, 0.03, 0.03, 0.05, 0.1,
    0.3, 0.6, 0.85, 1.0, 1.0, 0.95,
    0.9, 0.85, 0.8, 0.75, 0.7, 0.65,
    0.6, 0.5, 0.4, 0.35, 0.25, 0.15,
]


def _noise(base: float, pct: float = 0.12) -> float:
    """Add realistic noise to a base value."""
    return max(0.0, base * (1 + random.gauss(0, pct)))


def generate_platform_row(platform: str, hour: int, date: datetime) -> Dict[str, Any]:
    """Generate one hourly platform metrics row."""
    multiplier = HOURLY_MULTIPLIERS[hour]

    base_impressions = {"google": 12000, "meta": 18000, "microsoft": 4000}[platform]
    base_ctr = {"google": 0.045, "meta": 0.022, "microsoft": 0.038}[platform]
    base_cpc = {"google": 3.20, "meta": 1.85, "microsoft": 2.10}[platform]
    base_cvr = {"google": 0.062, "meta": 0.038, "microsoft": 0.055}[platform]

    impressions = int(_noise(base_impressions * multiplier, 0.15))
    clicks = int(impressions * _noise(base_ctr, 0.1))
    cost = round(clicks * _noise(base_cpc, 0.08), 2)
    conversions = int(clicks * _noise(base_cvr, 0.12))

    return {
        "date": date.strftime("%Y-%m-%d"),
        "hour": hour,
        "platform": platform,
        "impressions": impressions,
        "clicks": clicks,
        "cost": cost,
        "conversions": conversions,
        "cpc": round(cost / clicks, 4) if clicks > 0 else 0.0,
        "ctr": round(clicks / impressions, 4) if impressions > 0 else 0.0,
        "cvr": round(conversions / clicks, 4) if clicks > 0 else 0.0,
    }


def generate_call_center_row(hour: int, date: datetime, platform_cost: float) -> Dict[str, Any]:
    """Generate call center actual data matching the dashboard schema."""
    multiplier = HOURLY_MULTIPLIERS[hour]

    # Revenue per lead by type
    inbound_calls = int(_noise(8 * multiplier, 0.2))
    data_submits = int(_noise(15 * multiplier, 0.2))

    # Attribution rates from schema
    medium_income_verified_rate = _noise(0.42, 0.05)
    high_income_verified_carrier_rate = _noise(0.28, 0.06)

    # Revenue calculations
    call_revenue_per_call = _noise(85.0, 0.1)
    data_revenue_per_submit = _noise(22.0, 0.1)

    calls_revenue = round(inbound_calls * call_revenue_per_call, 2)
    data_revenue = round(data_submits * data_revenue_per_submit, 2)
    total_revenue = round(calls_revenue + data_revenue, 2)

    margin_dollar = round(total_revenue - platform_cost, 2)
    margin_pct = round(margin_dollar / total_revenue, 4) if total_revenue > 0 else 0.0

    return {
        "date": date.strftime("%Y-%m-%d"),
        "hour": hour,
        "calls_revenue": calls_revenue,
        "data_revenue": data_revenue,
        "total_revenue": total_revenue,
        "inbound_calls": inbound_calls,
        "data_submit_forms": data_submits,
        "medium_income_verified_rate": round(medium_income_verified_rate, 4),
        "high_income_verified_carrier_rate": round(high_income_verified_carrier_rate, 4),
        "margin_dollar": margin_dollar,
        "margin_pct": margin_pct,
    }


def generate_call_quality_row(hour: int, date: datetime) -> Dict[str, Any]:
    """Generate Call Quality Control data — medium income call ratio by duration."""
    multiplier = HOURLY_MULTIPLIERS[hour]
    total_calls = int(_noise(12 * multiplier, 0.18))

    # Ratio of medium income calls by duration bucket
    ratio_30s = round(_noise(0.35, 0.08), 4)
    ratio_60s = round(_noise(0.28, 0.08), 4)
    ratio_90s = round(_noise(0.20, 0.08), 4)

    return {
        "date": date.strftime("%Y-%m-%d"),
        "hour": hour,
        "total_calls": total_calls,
        "medium_income_call_ratio_30s": ratio_30s,
        "medium_income_call_ratio_60s": ratio_60s,
        "medium_income_call_ratio_90s": ratio_90s,
        "calls_30s": int(total_calls * ratio_30s),
        "calls_60s": int(total_calls * ratio_60s),
        "calls_90s": int(total_calls * ratio_90s),
    }


def generate_data_quality_row(hour: int, date: datetime) -> Dict[str, Any]:
    """Generate Data Quality Control — high income verified carrier data ratio."""
    return {
        "date": date.strftime("%Y-%m-%d"),
        "hour": hour,
        "high_income_verified_carrier_low_income_ratio": round(_noise(0.15, 0.1), 4),
        "high_income_verified_carrier_medium_income_ratio": round(_noise(0.32, 0.08), 4),
        "high_income_verified_carrier_high_income_ratio": round(_noise(0.53, 0.07), 4),
    }


def generate_attribution_rates_row(hour: int, date: datetime) -> Dict[str, Any]:
    """Generate Attribution Rates — inbound call, medium income verified, any data, high income with carrier."""
    return {
        "date": date.strftime("%Y-%m-%d"),
        "hour": hour,
        "inbound_call_rate": round(_noise(0.35, 0.06), 4),
        "medium_income_verified_rate": round(_noise(0.42, 0.05), 4),
        "any_data_rate": round(_noise(0.60, 0.04), 4),
        "high_income_verified_carrier_rate": round(_noise(0.28, 0.07), 4),
    }


def generate_full_day(date: datetime = None) -> Dict[str, List[Dict]]:
    """
    Generate a complete day of data for all tables and all platforms.
    Returns dict of {table_name: [rows]}
    """
    if date is None:
        date = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    platform_rows = []
    call_center_rows = []
    call_quality_rows = []
    data_quality_rows = []
    attribution_rows = []

    for hour in range(24):
        total_platform_cost = 0.0
        for platform in PLATFORMS:
            row = generate_platform_row(platform, hour, date)
            platform_rows.append(row)
            total_platform_cost += row["cost"]

        call_center_rows.append(generate_call_center_row(hour, date, total_platform_cost))
        call_quality_rows.append(generate_call_quality_row(hour, date))
        data_quality_rows.append(generate_data_quality_row(hour, date))
        attribution_rows.append(generate_attribution_rates_row(hour, date))

    return {
        "platform_metrics": platform_rows,
        "call_center": call_center_rows,
        "call_quality": call_quality_rows,
        "data_quality": data_quality_rows,
        "attribution_rates": attribution_rows,
    }


def generate_historical(days: int = 30) -> Dict[str, List[Dict]]:
    """Generate N days of historical data for seeding the DB."""
    result: Dict[str, List] = {
        "platform_metrics": [],
        "call_center": [],
        "call_quality": [],
        "data_quality": [],
        "attribution_rates": [],
    }
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    for d in range(days, 0, -1):
        day_data = generate_full_day(today - timedelta(days=d))
        for key in result:
            result[key].extend(day_data[key])
    return result


def generate_current_hour() -> Dict[str, List[Dict]]:
    """Generate data for the current hour only — used for real-time updates."""
    now = datetime.utcnow()
    hour = now.hour
    platform_rows = []
    total_cost = 0.0
    for platform in PLATFORMS:
        row = generate_platform_row(platform, hour, now)
        platform_rows.append(row)
        total_cost += row["cost"]

    return {
        "platform_metrics": platform_rows,
        "call_center": [generate_call_center_row(hour, now, total_cost)],
        "call_quality": [generate_call_quality_row(hour, now)],
        "data_quality": [generate_data_quality_row(hour, now)],
        "attribution_rates": [generate_attribution_rates_row(hour, now)],
    }
