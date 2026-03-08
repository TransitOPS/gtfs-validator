"""Tests for the notice system."""

from gtfs_validator.notices import Notice, NoticeContainer, Severity, SystemError


def test_notice_creation():
    n = Notice(code="test_code", severity=Severity.ERROR, fields={"a": 1})
    assert n.code == "test_code"
    assert n.severity == Severity.ERROR
    assert n.fields == {"a": 1}


def test_severity_ordering():
    assert Severity.INFO < Severity.WARNING < Severity.ERROR


def test_container_add_and_count():
    c = NoticeContainer()
    c.add(Notice("a", Severity.ERROR, {}))
    c.add(Notice("a", Severity.ERROR, {}))
    c.add(Notice("b", Severity.WARNING, {}))
    assert c.total == 3
    assert c.counts() == {("a", Severity.ERROR): 2, ("b", Severity.WARNING): 1}


def test_container_per_type_cap():
    c = NoticeContainer()
    c.MAX_PER_TYPE_AND_SEVERITY = 5
    for i in range(10):
        c.add(Notice("x", Severity.ERROR, {"i": i}))
    # Only 5 stored, but count tracks all 10.
    assert len(c.grouped_notices()[("x", Severity.ERROR)]) == 5
    assert c.counts()[("x", Severity.ERROR)] == 10
    assert c.total == 5  # stored total


def test_container_total_cap():
    c = NoticeContainer()
    c.MAX_TOTAL = 3
    for i in range(5):
        c.add(Notice(f"n{i}", Severity.INFO, {}))
    assert c.total == 3


def test_container_merge():
    c1 = NoticeContainer()
    c1.add(Notice("a", Severity.ERROR, {"v": 1}))
    c2 = NoticeContainer()
    c2.add(Notice("a", Severity.ERROR, {"v": 2}))
    c2.add(Notice("b", Severity.WARNING, {}))
    c1.merge(c2)
    assert c1.counts()[("a", Severity.ERROR)] == 2
    assert c1.counts()[("b", Severity.WARNING)] == 1
    assert c1.total == 3


def test_system_errors():
    c = NoticeContainer()
    assert not c.has_system_errors()
    c.add_system_error(SystemError(code="io_error", fields={"msg": "fail"}))
    assert c.has_system_errors()
    assert len(c.system_errors) == 1
