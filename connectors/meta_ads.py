"""
Meta (Facebook/Instagram) Ads connector — pulls hourly campaign performance
via the Marketing API Insights endpoint.

Setup:
    pip install facebook-business

Required env vars:
    META_APP_ID
    META_APP_SECRET
    META_ACCESS_TOKEN         (long-lived system-user token, not a short-lived one)
    META_AD_ACCOUNT_ID        (format: act_1234567890)

Notes: Meta's hourly breakdown is `hourly_stats_aggregated_by_advertiser_time_zone`,
returned as strings like "00:00:00 - 00:59:59" — we parse the hour out of that.
Conversions come back inside the `actions` array (one entry per action_type),
not as a single field, which is exactly the "multiple conversion actions"
harmonization problem the mapping layer solves.
"""
import os
from connectors.base import ConnectorResult, platform_metric_row, conversion_action_row, ConnectorAuthError, ConnectorAPIError

PLATFORM = "meta"

_REQUIRED_ENV = ["META_APP_ID", "META_APP_SECRET", "META_ACCESS_TOKEN", "META_AD_ACCOUNT_ID"]

# Which `actions[].action_type` values count as a conversion worth tracking.
# Extend this list to match what's actually configured for the ad account.
TRACKED_ACTION_TYPES = [
    "onsite_conversion.lead_grouped",
    "onsite_conversion.call_confirm",
    "offsite_conversion.fb_pixel_purchase",
    "lead",
    "purchase",
]


def _get_account():
    missing = [v for v in _REQUIRED_ENV if not os.getenv(v)]
    if missing:
        raise ConnectorAuthError(f"Meta Ads: missing env vars {missing}")
    try:
        from facebook_business.api import FacebookAdsApi
        from facebook_business.adobjects.adaccount import AdAccount
    except ImportError as e:
        raise ConnectorAPIError("facebook-business package not installed — pip install facebook-business") from e

    FacebookAdsApi.init(
        os.environ["META_APP_ID"], os.environ["META_APP_SECRET"], os.environ["META_ACCESS_TOKEN"]
    )
    return AdAccount(os.environ["META_AD_ACCOUNT_ID"])


def _hour_from_bucket(bucket: str) -> int:
    """'hourly_stats_aggregated_by_advertiser_time_zone' looks like '09:00:00 - 09:59:59'."""
    try:
        return int(bucket.split(":")[0])
    except Exception:
        return 0


def fetch_full_day(date: str) -> ConnectorResult:
    account = _get_account()
    result = ConnectorResult()

    fields = ["campaign_id", "campaign_name", "impressions", "clicks", "spend", "actions", "action_values"]
    params = {
        "time_range": {"since": date, "until": date},
        "level": "campaign",
        "breakdowns": ["hourly_stats_aggregated_by_advertiser_time_zone"],
        "limit": 500,
    }

    try:
        insights = account.get_insights(fields=fields, params=params)
        for row in insights:
            hour = _hour_from_bucket(row.get("hourly_stats_aggregated_by_advertiser_time_zone", "00:00:00 - 00:59:59"))
            impressions = int(row.get("impressions", 0))
            clicks = int(row.get("clicks", 0))
            spend = float(row.get("spend", 0.0))
            campaign_id = row.get("campaign_id")
            campaign_name = row.get("campaign_name")

            actions = row.get("actions", []) or []
            action_values = {a["action_type"]: float(a.get("value", 0)) for a in (row.get("action_values", []) or [])}
            tracked = [a for a in actions if a.get("action_type") in TRACKED_ACTION_TYPES]
            total_conversions = sum(float(a.get("value", 0)) for a in tracked)

            result.platform_metrics.append(
                platform_metric_row(date, hour, PLATFORM, impressions, clicks, spend, total_conversions,
                                     campaign_id=campaign_id, campaign_name=campaign_name)
            )
            for a in tracked:
                action_type = a["action_type"]
                result.conversion_actions.append(conversion_action_row(
                    date, hour, PLATFORM,
                    native_action_name=action_type,
                    conversions=float(a.get("value", 0)),
                    conversion_value=action_values.get(action_type, 0.0),
                    campaign_id=campaign_id, campaign_name=campaign_name,
                ))
    except Exception as e:
        raise ConnectorAPIError(f"Meta Ads fetch_full_day failed: {e}") from e

    return result


def fetch_hourly(date: str, hour: int) -> ConnectorResult:
    """Meta's API doesn't support filtering to a single hour server-side with
    this breakdown — pull the full day and filter client-side. Fine for an
    hourly cron since results are cached/cheap; swap for a smarter diff if
    Meta's rate limits become a problem at your account's call volume."""
    full_day = fetch_full_day(date)
    result = ConnectorResult()
    result.platform_metrics = [r for r in full_day.platform_metrics if r["hour"] == hour]
    result.conversion_actions = [r for r in full_day.conversion_actions if r["hour"] == hour]
    return result
