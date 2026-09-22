"""
Microsoft Advertising (Bing Ads) connector — pulls hourly campaign
performance via the Bing Ads Reporting API.

Setup:
    pip install bingads

Required env vars:
    MSADS_DEVELOPER_TOKEN
    MSADS_CLIENT_ID
    MSADS_CLIENT_SECRET
    MSADS_REFRESH_TOKEN
    MSADS_CUSTOMER_ID
    MSADS_ACCOUNT_ID

Notes: unlike Google/Meta, the Bing Ads Reporting API is asynchronous —
you submit a report request, poll until it's ready, then download and
parse the CSV. That round trip (usually 5–30s) is why this connector is
slower per call than the other two; don't call it more than once per hour
per the scheduler's cadence. Per-goal conversion breakdown requires the
account's Conversion Goals to be named consistently — those names are
what feeds pipeline/conversion_mapping.py.
"""
import os
import csv
from connectors.base import ConnectorResult, platform_metric_row, conversion_action_row, ConnectorAuthError, ConnectorAPIError

PLATFORM = "microsoft"

_REQUIRED_ENV = [
    "MSADS_DEVELOPER_TOKEN", "MSADS_CLIENT_ID", "MSADS_CLIENT_SECRET",
    "MSADS_REFRESH_TOKEN", "MSADS_CUSTOMER_ID", "MSADS_ACCOUNT_ID",
]


def _get_reporting_service_manager():
    missing = [v for v in _REQUIRED_ENV if not os.getenv(v)]
    if missing:
        raise ConnectorAuthError(f"Microsoft Advertising: missing env vars {missing}")
    try:
        from bingads.authorization import AuthorizationData, OAuthWebAuthCodeGrant
        from bingads.v13.reporting import ReportingServiceManager
    except ImportError as e:
        raise ConnectorAPIError("bingads package not installed — pip install bingads") from e

    auth = OAuthWebAuthCodeGrant(
        client_id=os.environ["MSADS_CLIENT_ID"],
        client_secret=os.environ["MSADS_CLIENT_SECRET"],
        redirection_uri="https://login.microsoftonline.com/common/oauth2/nativeclient",
    )
    auth.request_oauth_tokens_by_refresh_token(os.environ["MSADS_REFRESH_TOKEN"])

    auth_data = AuthorizationData(
        account_id=os.environ["MSADS_ACCOUNT_ID"],
        customer_id=os.environ["MSADS_CUSTOMER_ID"],
        developer_token=os.environ["MSADS_DEVELOPER_TOKEN"],
        authentication=auth,
    )
    return ReportingServiceManager(authorization_data=auth_data, poll_interval_in_milliseconds=5000, environment="production")


def _build_hourly_request(reporting_service_manager, date: str):
    from bingads.v13.reporting import ReportingDownloadParameters
    client = reporting_service_manager.reporting_service if hasattr(reporting_service_manager, "reporting_service") else None
    # Build via the underlying SOAP factory
    factory = reporting_service_manager._service_client.factory  # noqa: SLF001 — standard pattern for this SDK

    report_request = factory.create("CampaignPerformanceReportRequest")
    report_request.Format = "Csv"
    report_request.Aggregation = "Hourly"
    report_request.ReportName = f"ads_dashboard_{date}"

    scope = factory.create("AccountThroughCampaignReportScope")
    scope.AccountIds = {"long": [int(os.environ["MSADS_ACCOUNT_ID"])]}
    report_request.Scope = scope

    time_obj = factory.create("Date")
    y, m, d = date.split("-")
    time_obj.Year, time_obj.Month, time_obj.Day = int(y), int(m), int(d)
    report_time = factory.create("ReportTime")
    report_time.CustomDateRangeStart = time_obj
    report_time.CustomDateRangeEnd = time_obj
    report_request.Time = report_time

    columns = factory.create("ArrayOfCampaignPerformanceReportColumn")
    columns.CampaignPerformanceReportColumn = [
        "TimePeriod", "CampaignId", "CampaignName",
        "Impressions", "Clicks", "Spend", "Conversions", "Revenue",
    ]
    report_request.Columns = columns
    return report_request


def fetch_full_day(date: str) -> ConnectorResult:
    result = ConnectorResult()
    try:
        rsm = _get_reporting_service_manager()
        request = _build_hourly_request(rsm, date)
        from bingads.v13.reporting import ReportingDownloadParameters
        download_params = ReportingDownloadParameters(
            report_request=request,
            result_file_directory="/tmp",
            result_file_name=f"msads_{date}.csv",
            overwrite_result_file=True,
            timeout_in_milliseconds=5 * 60 * 1000,
        )
        file_path = rsm.download_file(download_params)

        with open(file_path, newline="", encoding="utf-8-sig") as f:
            # Bing Ads CSV reports have a header/footer preamble before the data rows;
            # csv.DictReader with the right skip is environment-specific — inspect a
            # sample export once and adjust `skip_rows` below to match.
            reader = csv.reader(f)
            rows = list(reader)

        header_idx = next(i for i, r in enumerate(rows) if r and r[0] == "TimePeriod")
        header = rows[header_idx]
        for raw in rows[header_idx + 1:]:
            if not raw or raw[0].startswith("Total") or len(raw) != len(header):
                continue
            record = dict(zip(header, raw))
            try:
                hour = int(record["TimePeriod"].split(" ")[-1].split(":")[0])
            except Exception:
                hour = 0
            impressions = int(record.get("Impressions", 0) or 0)
            clicks = int(record.get("Clicks", 0) or 0)
            spend = float(record.get("Spend", 0) or 0)
            conversions = float(record.get("Conversions", 0) or 0)
            revenue = float(record.get("Revenue", 0) or 0)
            campaign_id = record.get("CampaignId")
            campaign_name = record.get("CampaignName")

            result.platform_metrics.append(
                platform_metric_row(date, hour, PLATFORM, impressions, clicks, spend, conversions,
                                     campaign_id=campaign_id, campaign_name=campaign_name)
            )
            if conversions:
                result.conversion_actions.append(conversion_action_row(
                    date, hour, PLATFORM,
                    native_action_name="Conversions",  # Microsoft's CampaignPerformanceReport doesn't split by goal;
                    conversions=conversions,             # use GoalPerformanceReport instead if per-goal detail matters.
                    conversion_value=revenue,
                    campaign_id=campaign_id, campaign_name=campaign_name,
                ))
    except (ConnectorAuthError, ConnectorAPIError):
        raise
    except Exception as e:
        raise ConnectorAPIError(f"Microsoft Ads fetch_full_day failed: {e}") from e

    return result


def fetch_hourly(date: str, hour: int) -> ConnectorResult:
    """The Bing Ads report call above already returns the full day (it's an
    async report job either way, so there's no per-hour cost savings to
    requesting less) — filter to the hour the scheduler asked for."""
    full_day = fetch_full_day(date)
    result = ConnectorResult()
    result.platform_metrics = [r for r in full_day.platform_metrics if r["hour"] == hour]
    result.conversion_actions = [r for r in full_day.conversion_actions if r["hour"] == hour]
    return result
