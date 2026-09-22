"""
Shared contract for live ad-platform connectors.

Every connector (google_ads.py, meta_ads.py, microsoft_ads.py) exposes:

    fetch_hourly(date: str, hour: int) -> ConnectorResult
    fetch_full_day(date: str) -> ConnectorResult

...and returns data already normalized into the same shape ads_generator.py
produces, so pipeline/ingest.py and everything downstream (transforms,
Claude engine, dashboard) doesn't need to know whether it's looking at
synthetic or live data.

Design choice: each connector fails LOUD (raises) on auth/API errors rather
than silently returning empty data — scheduler.py decides whether to fall
back to synthetic data or surface the failure, but the connector itself
should never mask a broken credential as "zero spend today."
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ConnectorResult:
    platform_metrics: List[Dict[str, Any]] = field(default_factory=list)
    conversion_actions: List[Dict[str, Any]] = field(default_factory=list)

    def as_ingest_payload(self) -> Dict[str, List[Dict]]:
        """Shape expected by pipeline.ingest.ingest_all()."""
        return {
            "platform_metrics": self.platform_metrics,
            "conversion_actions": self.conversion_actions,
        }


def platform_metric_row(
    date: str, hour: int, platform: str,
    impressions: int, clicks: int, cost: float, conversions: float,
    campaign_id: str = None, campaign_name: str = None,
) -> Dict[str, Any]:
    """Build one normalized platform_metrics row with derived rate fields."""
    cpc = round(cost / clicks, 4) if clicks else 0.0
    ctr = round(clicks / impressions, 4) if impressions else 0.0
    cvr = round(conversions / clicks, 4) if clicks else 0.0
    return {
        "date": date, "hour": hour, "platform": platform,
        "campaign_id": campaign_id, "campaign_name": campaign_name,
        "impressions": int(impressions), "clicks": int(clicks),
        "cost": round(float(cost), 2), "conversions": round(float(conversions), 2),
        "cpc": cpc, "ctr": ctr, "cvr": cvr,
        "source": "live",
    }


def conversion_action_row(
    date: str, hour: int, platform: str, native_action_name: str,
    conversions: float, conversion_value: float,
    campaign_id: str = None, campaign_name: str = None,
) -> Dict[str, Any]:
    from pipeline.conversion_mapping import map_conversion_action
    return {
        "date": date, "hour": hour, "platform": platform,
        "campaign_id": campaign_id, "campaign_name": campaign_name,
        "native_action_name": native_action_name,
        "canonical_type": map_conversion_action(native_action_name, platform),
        "conversions": round(float(conversions), 2),
        "conversion_value": round(float(conversion_value), 2),
    }


class ConnectorAuthError(Exception):
    """Raised when credentials are missing or the platform rejects them."""


class ConnectorAPIError(Exception):
    """Raised on a non-auth API failure (rate limit, timeout, bad request)."""
