"""CLI entry point using click."""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import click

from gtfs_validator.config import RunResult, ValidationConfig
from gtfs_validator.runner import run


@click.command()
@click.option("-i", "--input", "input_path", type=click.Path(), default=None,
              help="Path to GTFS ZIP or directory.")
@click.option("-u", "--url", type=str, default=None,
              help="URL to GTFS archive.")
@click.option("-o", "--output_base", type=click.Path(), required=True,
              help="Output directory for reports.")
@click.option("-s", "--storage_directory", type=click.Path(), default=None,
              help="Disk path to save downloaded ZIP (only with --url).")
@click.option("-t", "--threads", type=int, default=1,
              help="Number of parallel threads.")
@click.option("-c", "--country_code", type=str, default=None,
              help="ISO 3166-1 alpha-2 country code.")
@click.option("-d", "--date", "date_for_validation", type=str, default=None,
              help="Validation date (YYYY-MM-DD).")
@click.option("-v", "--validation_report_name", type=str, default=None,
              help="Custom JSON report filename.")
@click.option("-r", "--html_report_name", type=str, default=None,
              help="Custom HTML report filename.")
@click.option("-e", "--system_errors_report_name", type=str, default=None,
              help="Custom system errors filename.")
@click.option("-p", "--pretty", is_flag=True, default=False,
              help="Pretty-print JSON output.")
@click.option("-n", "--export_notices_schema", is_flag=True, default=False,
              help="Export notice schema and exit.")
def main(
    input_path: str | None,
    url: str | None,
    output_base: str,
    storage_directory: str | None,
    threads: int,
    country_code: str | None,
    date_for_validation: str | None,
    validation_report_name: str | None,
    html_report_name: str | None,
    system_errors_report_name: str | None,
    pretty: bool,
    export_notices_schema: bool,
) -> None:
    """Validate a GTFS feed and generate reports."""

    # Handle --export_notices_schema short-circuit.
    if export_notices_schema:
        from gtfs_validator.notices import Severity
        _export_schema(Path(output_base))
        sys.exit(0)

    # Exactly one of --input or --url must be provided.
    if input_path and url:
        raise click.UsageError("Provide either --input or --url, not both.")
    if not input_path and not url:
        raise click.UsageError("Either --input or --url is required.")

    # --storage_directory only valid with --url.
    if storage_directory and not url:
        raise click.UsageError("--storage_directory is only valid with --url.")

    # Validate threads.
    if threads < 1:
        raise click.UsageError("--threads must be >= 1.")

    # Parse and validate date.
    parsed_date = date.today()
    if date_for_validation:
        try:
            parsed_date = date.fromisoformat(date_for_validation)
        except ValueError:
            raise click.UsageError(
                f"Invalid date format: {date_for_validation!r}. Use YYYY-MM-DD."
            )

    # Validate and normalise country code.
    cc = "ZZ"
    if country_code:
        if not re.fullmatch(r"[A-Z]{2}", country_code.upper()):
            raise click.UsageError(
                f"Invalid country code: {country_code!r}. Use ISO 3166-1 alpha-2."
            )
        cc = country_code.upper()

    gtfs_source = input_path or url
    assert gtfs_source is not None  # guaranteed by checks above

    config = ValidationConfig(
        gtfs_source=gtfs_source,
        output_directory=Path(output_base),
        storage_directory=Path(storage_directory) if storage_directory else None,
        validation_report_name=validation_report_name or "report.json",
        html_report_name=html_report_name or "report.html",
        system_errors_report_name=system_errors_report_name or "system_errors.json",
        num_threads=threads,
        country_code=cc,
        date_for_validation=parsed_date,
        pretty_json=pretty,
    )

    result = run(config)
    sys.exit(result.value)


def _export_schema(output_dir: Path) -> None:
    """Export notice schema to JSON and exit."""
    import json

    # Build schema from known notice codes.
    # For now, output a placeholder until all notices are formally registered.
    output_dir.mkdir(parents=True, exist_ok=True)
    schema: list[dict[str, object]] = []
    (output_dir / "notice_schema.json").write_text(json.dumps(schema, indent=2))


if __name__ == "__main__":
    main()
