"""Validation context passed to all validators."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ValidationContext:
    """Immutable context created once before validator dispatch."""

    country_code: str
    date_for_validation: date
