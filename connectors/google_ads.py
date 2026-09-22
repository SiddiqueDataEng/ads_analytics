"""
Google Ads connector — pulls hourly campaign performance via the Google Ads
Query Language (GAQL) reporting API.

Setup:
    pip install google-ads
    Create google-ads.yaml (or set env vars below) with:
        developer_token, client_id, client_secret, refresh_token, login_customer_id

Required env vars (or put them in a google-ads.yaml and set GOOGLE_ADS_CONFIG_PATH):
    GOOGLE_ADS_DEVELOPER_TOKEN
    GOOGLE_ADS_CLIENT_ID
    GOOGLE_ADS_CLIENT_SECRET
    GOOGLE_ADS_REFRESH_TOKEN
    GOOGLE_ADS_LOGIN_CUSTOMER_ID   (MCC ID, digits only, no dashes)
    GOOGLE_ADS_CUSTOMER_ID         (the account being reported on)

Notes on hourly data: Google Ads' `segments.hour` is only populated on
today's data (and only via certain report types) — historical hour-level
data beyond ~90 days is not retained. For MTD/WoW history, query daily
(no segments.hour) and store it under hour=0, or run this hourly and let
your own DB accumulate the history going forward.
"""
import os
from datetime import datetime
from connectors.base import ConnectorResult, platform_metric_row, conversion_action_row, ConnectorAuthError, ConnectorAPIError

PLATFORM = "google"

_REQUIRED_ENV = [
    "GOOGLE_ADS_DEVELOPER_TOKEN", "GOOGLE_ADS_CLIENT_ID", "GOOGLE_ADS_CLIENT_SECRET",
    "GOOGLE_ADS_REFRESH_TOKEN", "GOOGLE_ADS_CUSTOMER_ID",
]


def _get_client():
    missing = [v for v in _REQUIRED_ENV if not os.getenv(v)]
    if missing:
        raise ConnectorAuthError(f"Google Ads: missing env vars {missing}")
    try:
        from google.ads.googleads.client import GoogleAdsClient
    except ImportError as e:
        raise ConnectorAPIError("google-ads package not installed — pip install google-ads") from e

    config = {
        "developer_token": os.environ["GOOGLE_ADS_DEVELOPER_TOKEN"],
        "client_id": os.environ["GOOGLE_ADS_CLIENT_ID"],
        "client_secret": os.environ["GOOGLE_ADS_CLIENT_SECRET"],
        "refresh_token": os.environ["GOOGLE_ADS_REFRESH_TOKEN"],
        "use_proto_plus": True,
    }
    if os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID"):
        config["login_customer_id"] = os.environ["GOOGLE_ADS_LOGIN_CUSTOMER_ID"]
    return GoogleAdsClient.load_from_dict(config)


_GAQL_HOURLY = """
    SELECT
      campaign.id, campaign.name,
      segments.hour, segments.date,
      metrics.impressions, metrics.clicks, metrics.cost_micros,
      metrics.conversions, metrics.conversions_value
    FROM campaign
    WHERE segments.date = '{date}'
      AND segments.hour = {hour}
"""

_GAQL_CONVERSION_ACTIONS = """
    SELECT
      campaign.id, campaign.name, segments.conversion_action_name,
      segments.hour, metrics.conversions, metrics.conversions_value
    FROM campaign
    WHERE segments.date = '{date}' AND segments.hour = {hour}
"""


def fetch_hourly(date: str, hour: int) -> ConnectorResult:
    client = _get_client()
    customer_id = os.environ["GOOGLE_ADS_CUSTOMER_ID"]
    ga_service = client.get_service("GoogleAdsService")
    result = ConnectorResult()

    try:
        rows = ga_service.search(customer_id=customer_id, query=_GAQL_HOURLY.format(date=date, hour=hour))
        agg = {}  # campaign_id -> totals, then a platform-level rollup row
        impr = clicks = conv = 0
        cost_micros = 0
        for row in rows:
            impr += row.metrics.impressions
            clicks += row.metrics.clicks
            cost_micros += row.metrics.cost_micros
            conv += row.metrics.conversions
        result.platform_metrics.append(
            platform_metric_row(date, hour, PLATFORM, impr, clicks, cost_micros / 1_000_000, conv)
        )

        conv_rows = ga_service.search(customer_id=customer_id, query=_GAQL_CONVERSION_ACTIONS.format(date=date, hour=hour))
        for row in conv_rows:
            if row.metrics.conversions == 0:
                continue
            result.conversion_actions.append(conversion_action_row(
                date, hour, PLATFORM,
                native_action_name=row.segments.conversion_action_name,
                conversions=row.metrics.conversions,
                conversion_value=row.metrics.conversions_value,
                campaign_id=str(row.campaign.id), campaign_name=row.campaign.name,
            ))
    except Exception as e:
        raise ConnectorAPIError(f"Google Ads fetch_hourly failed: {e}") from e

    return result


def fetch_full_day(date: str) -> ConnectorResult:
    """Loop all 24 hours. For historical (non-today) dates, prefer a single
    daily query without segments.hour — Google doesn't retain hourly history."""
    combined = ConnectorResult()
    is_today = date == datetime.utcnow().strftime("%Y-%m-%d")
    hours = range(24) if is_today else [0]  # collapse to one row per day for historical dates
    for h in hours:
        r = fetch_hourly(date, h)
        combined.platform_metrics.extend(r.platform_metrics)
        combined.conversion_actions.extend(r.conversion_actions)
    return combined
