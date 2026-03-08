"""Report generation: report.json, system_errors.json, and HTML."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


def _to_file_uri(source: str) -> str:
    """Convert a local path to an absolute file:// URI.

    Paths that are already URIs (http://, https://, file://) are returned
    unchanged.
    """
    if source.startswith(("http://", "https://", "file://")):
        return source
    return Path(source).resolve().as_uri()

import polars as pl

from gtfs_validator.config import ValidationConfig
from gtfs_validator.notices import NoticeContainer, Severity


def generate_reports(
    config: ValidationConfig,
    feed: dict[str, pl.DataFrame],
    notices: NoticeContainer,
    features: list[str],
    validation_time_seconds: float,
) -> None:
    """Write report.json, system_errors.json, and report.html."""
    config.output_directory.mkdir(parents=True, exist_ok=True)

    report = _build_validation_report(
        config, feed, notices, features, validation_time_seconds,
    )
    _write_json(
        config.output_directory / config.validation_report_name,
        report,
        config.pretty_json,
    )

    sys_errors = _build_system_errors_report(notices)
    _write_json(
        config.output_directory / config.system_errors_report_name,
        sys_errors,
        config.pretty_json,
    )

    _write_html(
        config.output_directory / config.html_report_name,
        report,
        config,
    )


# ---------------------------------------------------------------------------
# report.json
# ---------------------------------------------------------------------------


def _build_validation_report(
    config: ValidationConfig,
    feed: dict[str, pl.DataFrame],
    notices: NoticeContainer,
    features: list[str],
    elapsed: float,
) -> dict[str, Any]:
    from gtfs_validator import __version__

    summary: dict[str, Any] = {
        "validatorVersion": __version__,
        "validatedAt": datetime.now(timezone.utc).isoformat(),
        "gtfsInput": _to_file_uri(config.gtfs_source),
        "threads": config.num_threads,
        "outputDirectory": str(config.output_directory),
        "systemErrorsReportName": config.system_errors_report_name,
        "validationReportName": config.validation_report_name,
        "htmlReportName": config.html_report_name,
        "countryCode": config.country_code,
        "dateForValidation": config.date_for_validation.isoformat(),
    }

    # Feed info.
    fi = feed.get("feed_info.txt")
    if fi is not None and fi.height > 0:
        row = fi.row(0, named=True)
        summary["feedInfo"] = {
            k: v for k, v in {
                "publisherName": row.get("feed_publisher_name"),
                "publisherUrl": row.get("feed_publisher_url"),
                "feedLanguage": row.get("feed_lang"),
                "feedStartDate": row.get("feed_start_date"),
                "feedEndDate": row.get("feed_end_date"),
            }.items() if v is not None
        }

    # Agencies.
    ag = feed.get("agency.txt")
    if ag is not None and ag.height > 0:
        agencies = []
        for row in ag.iter_rows(named=True):
            agencies.append({
                k: v for k, v in {
                    "name": row.get("agency_name"),
                    "url": row.get("agency_url"),
                    "phone": row.get("agency_phone"),
                    "email": row.get("agency_email"),
                }.items() if v is not None
            })
        summary["agencies"] = agencies

    # Files present.
    summary["files"] = sorted(
        fname for fname, df in feed.items() if df.height > 0
    )

    # Entity counts.
    summary["counts"] = _entity_counts(feed)

    # Features.
    summary["gtfsFeatures"] = features

    # Timing.
    summary["validationTimeSeconds"] = round(elapsed, 3)

    # Notices.
    notice_groups = _group_notices(notices)

    return {"summary": summary, "notices": notice_groups}


def _entity_counts(feed: dict[str, pl.DataFrame]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, label in [
        ("shapes.txt", "Shapes"),
        ("stops.txt", "Stops"),
        ("routes.txt", "Routes"),
        ("trips.txt", "Trips"),
        ("agency.txt", "Agencies"),
    ]:
        df = feed.get(key)
        if df is not None:
            counts[label] = df.height

    # Blocks = distinct block_id values from trips.txt.
    trips = feed.get("trips.txt")
    if trips is not None and "block_id" in trips.columns:
        counts["Blocks"] = trips.select("block_id").drop_nulls().n_unique()

    return counts


def _group_notices(notices: NoticeContainer) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    all_notices = notices.grouped_notices()
    all_counts = notices.counts()

    for key in sorted(all_notices, key=lambda k: (k[1].value, k[0])):
        code, severity = key
        sample = all_notices[key]
        total = all_counts.get(key, len(sample))
        export_limit = notices.MAX_EXPORTS_PER_TYPE
        groups.append({
            "code": code,
            "severity": severity.name,
            "totalNotices": total,
            "sampleNotices": [n.fields for n in sample[:export_limit]],
        })

    return groups


# ---------------------------------------------------------------------------
# system_errors.json
# ---------------------------------------------------------------------------


def _build_system_errors_report(notices: NoticeContainer) -> dict[str, Any]:
    errors = notices.system_errors
    if not errors:
        return {"notices": []}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for err in errors:
        grouped.setdefault(err.code, []).append(err.fields)

    out: list[dict[str, Any]] = []
    for code, samples in grouped.items():
        out.append({
            "code": code,
            "severity": "ERROR",
            "totalNotices": len(samples),
            "sampleNotices": samples[:NoticeContainer.MAX_EXPORTS_PER_TYPE],
        })

    return {"notices": out}


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------


class _DateEncoder(json.JSONEncoder):
    def default(self, o: object) -> object:
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        return super().default(o)


def _write_json(path: Path, data: Any, pretty: bool) -> None:
    indent = 2 if pretty else None
    path.write_text(json.dumps(data, cls=_DateEncoder, indent=indent))


# ---------------------------------------------------------------------------
# HTML report (minimal Jinja2 template)
# ---------------------------------------------------------------------------


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>GTFS Validation Report</title>
<style>
body{font-family:sans-serif;margin:2em}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #ccc;padding:6px 10px;text-align:left}
th{background:#f5f5f5}
.error{color:#c00}.warning{color:#c80}.info{color:#08c}
</style></head>
<body>
<h1>GTFS Validation Report</h1>
<h2>Summary</h2>
<table>
<tr><th>Validator Version</th><td>{{ summary.validatorVersion }}</td></tr>
<tr><th>Validated At</th><td>{{ summary.validatedAt }}</td></tr>
<tr><th>GTFS Input</th><td>{{ summary.gtfsInput }}</td></tr>
<tr><th>Country Code</th><td>{{ summary.countryCode }}</td></tr>
<tr><th>Validation Date</th><td>{{ summary.dateForValidation }}</td></tr>
<tr><th>Time (seconds)</th><td>{{ summary.validationTimeSeconds }}</td></tr>
</table>
{% if summary.counts %}
<h2>Entity Counts</h2>
<table>
{% for label, count in summary.counts.items() %}
<tr><td>{{ label }}</td><td>{{ count }}</td></tr>
{% endfor %}
</table>
{% endif %}
{% if summary.gtfsFeatures %}
<h2>Features Detected</h2>
<ul>{% for f in summary.gtfsFeatures %}<li>{{ f }}</li>{% endfor %}</ul>
{% endif %}
<h2>Notices</h2>
{% if notices %}
<table>
<tr><th>Code</th><th>Severity</th><th>Total</th></tr>
{% for n in notices %}
<tr><td>{{ n.code }}</td>
<td class="{{ n.severity | lower }}">{{ n.severity }}</td>
<td>{{ n.totalNotices }}</td></tr>
{% endfor %}
</table>
{% else %}
<p>No notices.</p>
{% endif %}
</body></html>
"""


def _write_html(path: Path, report: dict[str, Any], config: ValidationConfig) -> None:
    try:
        from jinja2 import Environment
    except ImportError:
        # If Jinja2 is not installed, skip HTML generation.
        return

    env = Environment(autoescape=True)
    template = env.from_string(_HTML_TEMPLATE)
    html = template.render(
        summary=report["summary"],
        notices=report["notices"],
        config=config,
    )
    path.write_text(html)
