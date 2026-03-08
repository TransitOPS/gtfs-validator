"""Notice system: Notice, SystemError, Severity, and NoticeContainer."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class Severity(IntEnum):
    INFO = 0
    WARNING = 1
    ERROR = 2


@dataclass(frozen=True)
class Notice:
    """A single validation finding.

    ``code`` is a snake_case identifier (e.g. ``"foreign_key_violation"``).
    ``fields`` carries context serialized to JSON in the report.
    """

    code: str
    severity: Severity
    fields: dict[str, Any]


@dataclass(frozen=True)
class SystemError:
    """An internal error (IO failure, validator crash, etc.).

    Always severity ERROR.  Goes to ``system_errors.json``, never ``report.json``.
    """

    code: str
    fields: dict[str, Any]
    severity: Severity = Severity.ERROR


class NoticeContainer:
    """Accumulates notices and system errors with capacity limits.

    Not thread-safe — each thread must own a private instance and merge
    into the main container after completion.
    """

    MAX_PER_TYPE_AND_SEVERITY: int = 100_000
    MAX_TOTAL: int = 10_000_000
    MAX_EXPORTS_PER_TYPE: int = 1_000

    def __init__(self) -> None:
        self._notices: dict[tuple[str, Severity], list[Notice]] = defaultdict(list)
        self._counts: dict[tuple[str, Severity], int] = defaultdict(int)
        self._total: int = 0
        self._system_errors: list[SystemError] = []

    def add(self, notice: Notice) -> None:
        key = (notice.code, notice.severity)
        self._counts[key] += 1
        if self._total >= self.MAX_TOTAL:
            return
        if len(self._notices[key]) >= self.MAX_PER_TYPE_AND_SEVERITY:
            return
        self._notices[key].append(notice)
        self._total += 1

    def add_all(self, notices: list[Notice]) -> None:
        for n in notices:
            self.add(n)

    def add_system_error(self, error: SystemError) -> None:
        self._system_errors.append(error)

    def merge(self, other: NoticeContainer) -> None:
        """Merge another container into this one (post-thread-completion)."""
        for key, notices in other._notices.items():
            incoming_total = other._counts.get(key, 0)
            existing_total = self._counts[key]
            for notice in notices:
                self.add(notice)
            # Preserve aggregate totals exactly after capped sample merge.
            self._counts[key] = existing_total + incoming_total
        self._system_errors.extend(other._system_errors)

    def has_system_errors(self) -> bool:
        return len(self._system_errors) > 0

    @property
    def system_errors(self) -> list[SystemError]:
        return self._system_errors

    def grouped_notices(self) -> dict[tuple[str, Severity], list[Notice]]:
        return dict(self._notices)

    def counts(self) -> dict[tuple[str, Severity], int]:
        return dict(self._counts)

    @property
    def total(self) -> int:
        return self._total
