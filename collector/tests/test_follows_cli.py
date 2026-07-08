from __future__ import annotations

from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from kenya_monitor.cli import app

runner = CliRunner()


def test_follows_without_handle():
    with patch(
        "kenya_monitor.scheduler.run_follows_once",
        new_callable=AsyncMock,
        return_value={"follow_edges": 0},
    ) as run:
        result = runner.invoke(app, ["follows"])
        assert result.exit_code == 0, result.stdout + result.stderr
        run.assert_awaited_once_with(handles=None, limit=500, max_accounts=30, top_suspicious=None)


def test_follows_with_explicit_handle():
    with patch(
        "kenya_monitor.scheduler.run_follows_once",
        new_callable=AsyncMock,
        return_value={"follow_edges": 12},
    ) as run:
        result = runner.invoke(app, ["follows", "--handle", "WilliamsRuto", "--limit", "10"])
        assert result.exit_code == 0, result.stdout + result.stderr
        run.assert_awaited_once_with(
            handles=["WilliamsRuto"],
            limit=10,
            max_accounts=30,
            top_suspicious=None,
        )
        assert "follow_edges" in result.stdout


def test_follows_top_suspicious():
    with patch(
        "kenya_monitor.scheduler.run_follows_once",
        new_callable=AsyncMock,
        return_value={"follow_edges": 100, "accounts": 1000},
    ) as run:
        result = runner.invoke(app, ["follows", "--top-suspicious", "1000"])
        assert result.exit_code == 0, result.stdout + result.stderr
        run.assert_awaited_once_with(
            handles=None,
            limit=500,
            max_accounts=1000,
            top_suspicious=1000,
        )
