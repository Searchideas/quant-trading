import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from broker.telegram import (
    claim_pending,
    execute_pending,
    flush_queued,
    format_report,
    is_approval,
    is_rejection,
    record_sleeve,
    session_phase,
)


def test_approval_words():
    assert is_approval("approve")
    assert is_approval("Execute")
    assert is_approval("/execute")
    assert not is_approval("approved later")
    assert is_rejection("reject")


def test_sleeve_history_compounds_from_the_first_day(tmp_path: Path):
    path = tmp_path / "sleeve_history.csv"
    first = record_sleeve(path, "2026-10-07", 100.0)
    second = record_sleeve(path, "2026-10-08", 110.0)
    assert first["total_return"] == 0
    assert second["daily_return"] == pytest.approx(0.1)
    assert second["total_return"] == pytest.approx(0.1)
    assert second["since"] == "2026-10-07"


def test_a_withdrawal_stays_out_of_the_return(tmp_path: Path):
    path = tmp_path / "sleeve_history.csv"
    record_sleeve(path, "2026-10-08", 15163.06, 0.0)
    today = record_sleeve(path, "2026-10-09", 14786.12, 455.74)
    assert today["daily_return"] == pytest.approx(15241.86 / 15163.06 - 1)
    assert today["total_return"] == pytest.approx(15241.86 / 15163.06 - 1)


def test_report_lists_weights_and_does_not_claim_orders_were_sent():
    plan = pd.DataFrame(
        [
            {
                "ticker": "CRM",
                "weight": 0.25,
                "target_shares": 20,
                "current_shares": 0,
            }
        ]
    )
    orders = pd.DataFrame([{"action": "BUY", "delta_shares": 20, "tiger_symbol": "CRM"}])
    text = format_report(
        {"currency": "USD", "sleeve": 17698.21, "plan": plan, "orders": orders},
        {"daily_return": float("nan"), "total_return": 0.0, "since": "2026-10-07"},
        "2026-10-07",
    )
    assert "CRM 25.0% -> 20 shares" in text
    assert "BUY 20 CRM" in text
    assert "Reply approve" in text
    assert "Excluded from this return and from orders: VOO, TSM, GLD, DBS, OCBC." in text


def test_wednesday_morning_in_singapore_is_before_the_us_open():
    moment = datetime(2026, 10, 7, 8, 57, tzinfo=ZoneInfo("Asia/Singapore"))
    assert session_phase("2026-10-07", moment) == "early"


def test_session_is_still_open_after_midnight_in_singapore():
    moment = datetime(2026, 10, 8, 1, 0, tzinfo=ZoneInfo("Asia/Singapore"))
    assert session_phase("2026-10-07", moment) == "open"


def _pending(tmp_path: Path, monkeypatch, status: str) -> Path:
    monkeypatch.setattr("broker.telegram.pending_path", lambda market="us": tmp_path / "pending.json")
    path = tmp_path / "pending.json"
    path.write_text(
        json.dumps(
            {
                "date": "2026-10-07",
                "status": status,
                "currency": "USD",
                "orders": [{"action": "BUY", "delta_shares": 1, "tiger_symbol": "CRM"}],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_approve_before_the_open_holds_the_plan(tmp_path: Path, monkeypatch):
    path = _pending(tmp_path, monkeypatch, "failed")
    monkeypatch.setattr("broker.telegram.session_phase", lambda plan_day, moment=None: "early")
    called = []
    monkeypatch.setattr("broker.tiger_client.place_orders", lambda *args, **kwargs: called.append(args))
    text = execute_pending("2026-10-07")
    assert "Nothing was sent" in text
    assert called == []
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "queued"


def test_queued_plan_is_sent_when_the_session_is_open(tmp_path: Path, monkeypatch):
    path = _pending(tmp_path, monkeypatch, "queued")
    monkeypatch.setattr("broker.telegram.session_phase", lambda plan_day, moment=None: "open")
    monkeypatch.setattr("broker.tiger_client.trade_client", lambda: object())

    def _place(client, orders, currency):
        assert currency == "USD"
        return [{"action": "BUY", "quantity": 1, "tiger_symbol": "CRM"}]

    monkeypatch.setattr("broker.tiger_client.place_orders", _place)
    text = execute_pending("2026-10-07")
    assert text == "Sent BUY 1 CRM"
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "sent"


def test_flush_waits_until_the_open(tmp_path: Path, monkeypatch):
    path = _pending(tmp_path, monkeypatch, "queued")
    monkeypatch.setattr("broker.telegram.session_phase", lambda plan_day, moment=None: "early")
    assert flush_queued() is None
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "queued"


def test_partial_send_is_resumed_without_repeating_filled_orders(tmp_path: Path, monkeypatch):
    path = _pending(tmp_path, monkeypatch, "failed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["orders"] = [
        {"action": "SELL", "delta_shares": -6, "tiger_symbol": "AMZN"},
        {"action": "BUY", "delta_shares": 18, "tiger_symbol": "CRM"},
    ]
    payload["submitted"] = [{"action": "SELL", "quantity": 6, "tiger_symbol": "AMZN"}]
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr("broker.telegram.session_phase", lambda plan_day, moment=None: "open")
    monkeypatch.setattr("broker.tiger_client.trade_client", lambda: object())

    def _place(client, orders, currency):
        assert list(orders["tiger_symbol"]) == ["CRM"]
        return [{"action": "BUY", "quantity": 18, "tiger_symbol": "CRM"}]

    monkeypatch.setattr("broker.tiger_client.place_orders", _place)
    text = execute_pending("2026-10-07")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert text == "Sent BUY 18 CRM"
    assert saved["status"] == "sent"
    assert [row["tiger_symbol"] for row in saved["submitted"]] == ["AMZN", "CRM"]


def test_claim_pending_only_once_and_only_today(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("broker.telegram.pending_path", lambda market="us": tmp_path / "pending.json")
    payload = {"date": "2026-10-07", "status": "pending", "currency": "USD", "orders": [{"action": "BUY"}]}
    (tmp_path / "pending.json").write_text(json.dumps(payload), encoding="utf-8")
    assert claim_pending("2026-10-06") is None
    claimed = claim_pending("2026-10-07")
    assert claimed is not None
    assert claim_pending("2026-10-07") is None
