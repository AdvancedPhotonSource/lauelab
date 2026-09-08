# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Shared pytest configuration.

Native availability is decided from the *imported* ``lauelab`` package
(installed or editable), never from a path inside the repository, so a build
that fails to ship a native file shows up as a skip rather than a silent pass.

``--require-native`` (used by CI) turns that protection into a hard gate:
the session fails if ``liblaue.so`` is missing from the installed package,
and it fails if any test skipped for a reason other than the GPU executable
being absent.
"""

from __future__ import annotations

from importlib import resources
import pytest

GPU_SKIP_REASON = "GPU reconstruction executable not available"
TWIN2_SKIP_REASON = "Twin2 wire-scan fixture not available"
# Valgrind and the C compiler ship in environment.yml, so their absence is a
# broken environment, not an expected skip.
ALLOWED_SKIP_REASONS = (GPU_SKIP_REASON, TWIN2_SKIP_REASON)


@pytest.fixture
def frozen_results_clock(monkeypatch):
    from datetime import datetime
    import lauelab._hdf5 as hdf5

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 4, tzinfo=tz)

    monkeypatch.setattr(hdf5, "datetime", FrozenDatetime)


def native_file_available(package: str, name: str) -> bool:
    try:
        return (resources.files(package) / name).is_file()
    except (ModuleNotFoundError, TypeError):
        return False


LIBLAUE_AVAILABLE = native_file_available("lauelab.indexing.bin", "liblaue.so")

requires_liblaue = pytest.mark.skipif(
    not LIBLAUE_AVAILABLE,
    reason="liblaue.so is not present in the installed lauelab package",
)


def pytest_addoption(parser):
    parser.addoption(
        "--require-native",
        action="store_true",
        default=False,
        help="fail the session if liblaue.so is missing or if any test skips for a reason not in the allowed list",
    )
    parser.addoption(
        "--regenerate-results-schema",
        action="store_true",
        default=False,
        help="rewrite tests/data/results/layout_synthetic.txt from the current writer before comparing",
    )


def pytest_sessionstart(session):
    if session.config.getoption("--require-native") and not LIBLAUE_AVAILABLE:
        raise pytest.UsageError(
            "--require-native: liblaue.so is not present in the installed lauelab package"
        )


def _unexpected_skips(config):
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    unexpected = []
    for report in reporter.stats.get("skipped", []) if reporter else []:
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        reason = reason.removeprefix("Skipped: ")
        if not any(reason.startswith(allowed) for allowed in ALLOWED_SKIP_REASONS):
            unexpected.append((report.nodeid, reason))
    return unexpected


def pytest_sessionfinish(session, exitstatus):
    if not session.config.getoption("--require-native"):
        return
    unexpected = _unexpected_skips(session.config)
    if unexpected:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        reporter.write_line("")
        reporter.write_line("--require-native: tests skipped for a reason that is not allowed:", red=True)
        for nodeid, reason in unexpected:
            reporter.write_line(f"  {nodeid}: {reason}", red=True)
        session.exitstatus = 1
