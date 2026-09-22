"""
Fans out to all three platform connectors and merges results into the
shape pipeline.ingest.ingest_all() expects.

Failure isolation: if Microsoft's API is down but Google and Meta succeed,
we still ingest what we got and log which platform failed — one platform's
outage shouldn't blank out the whole dashboard. main.py / scheduler.py can
inspect `LiveFetchReport.errors` to raise an alert (e.g. post to Slack).
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

from connectors import google_ads, meta_ads, microsoft_ads
from connectors.base import ConnectorAuthError, ConnectorAPIError

CONNECTORS = {
    "google": google_ads,
    "meta": meta_ads,
    "microsoft": microsoft_ads,
}


@dataclass
class LiveFetchReport:
    payload: Dict[str, List[Dict]] = field(default_factory=lambda: {"platform_metrics": [], "conversion_actions": []})
    succeeded: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def all_failed(self) -> bool:
        return len(self.errors) == len(CONNECTORS)


def fetch_all_hourly(date: str, hour: int) -> LiveFetchReport:
    report = LiveFetchReport()
    for platform, module in CONNECTORS.items():
        try:
            result = module.fetch_hourly(date, hour)
            report.payload["platform_metrics"].extend(result.platform_metrics)
            report.payload["conversion_actions"].extend(result.conversion_actions)
            report.succeeded.append(platform)
        except (ConnectorAuthError, ConnectorAPIError) as e:
            report.errors[platform] = str(e)
        except Exception as e:  # noqa: BLE001 — a connector bug shouldn't take down the other two platforms
            report.errors[platform] = f"unexpected error: {e}"
    return report


def fetch_all_full_day(date: str) -> LiveFetchReport:
    report = LiveFetchReport()
    for platform, module in CONNECTORS.items():
        try:
            result = module.fetch_full_day(date)
            report.payload["platform_metrics"].extend(result.platform_metrics)
            report.payload["conversion_actions"].extend(result.conversion_actions)
            report.succeeded.append(platform)
        except (ConnectorAuthError, ConnectorAPIError) as e:
            report.errors[platform] = str(e)
        except Exception as e:  # noqa: BLE001
            report.errors[platform] = f"unexpected error: {e}"
    return report
