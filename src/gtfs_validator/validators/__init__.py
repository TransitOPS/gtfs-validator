"""Explicit validator registry for cross-table business-logic validators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import polars as pl

    from gtfs_validator.context import ValidationContext
    from gtfs_validator.notices import Notice

ValidatorFn = Callable[
    ["dict[str, pl.DataFrame]", "ValidationContext"],
    "list[Notice]",
]


@dataclass(frozen=True)
class ValidatorEntry:
    """A registered cross-table validator with its dependency metadata."""

    name: str
    fn: ValidatorFn
    requires: list[str]


# Validators are added here as they are implemented.
# Order matters — validators run in registry order.
VALIDATOR_REGISTRY: list[ValidatorEntry] = []
