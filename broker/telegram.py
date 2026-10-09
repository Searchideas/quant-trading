"""Morning Telegram report, and order placement only after an explicit approve.

    python -m broker.telegram report
    python -m broker.telegram listen
    python -m broker.telegram execute

Credentials live in telegram.env (not committed):

    TELEGRAM_BOT_TOKEN=123:abc
    TELEGRAM_CHAT_ID=999
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, time as clock_time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from broker.plan import EXCLUDED_TICKERS
from paths import get_paths

SGT = ZoneInfo("Asia/Singapore")
ET = ZoneInfo("America/New_York")
ENV_PATH = ROOT / "telegram.env"
APPROVE = {"approve", "/approve", "execute", "/execute"}
REJECT = {"reject", "/reject"}
_PLAN_LOCK = threading.RLock()
HELD_MESSAGE = (
    "Approved. Nothing was sent.\n"
    "Tiger accepts market orders only from 9:30am to 4:00pm ET (9:30pm to 4:00am SGT).\n"
    "This plan is saved and will be sent when that session opens."
)


def now_sgt() -> datetime:
    return datetime.now(SGT)


def load_env(path: Path = ENV_PATH) -> dict[str, str]:
    if not path.is_file():
        raise SystemExit(
            "telegram.env is missing. Create a bot with @BotFather, put "
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in telegram.env, then retry."
        )
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    if not values.get("TELEGRAM_BOT_TOKEN") or not values.get("TELEGRAM_CHAT_ID"):
        raise SystemExit("telegram.env needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
    return values


def record_sleeve(
    history_path: Path, day: str, sleeve: float, withdrawn: float = 0.0
) -> dict[str, float]:
    """Append today's sleeve value. Withdrawals are left out of the return."""
    row = {"date": day, "sleeve_value": sleeve, "withdrawn": withdrawn}
    if history_path.exists():
        history = pd.read_csv(history_path)
        history["date"] = history["date"].astype(str)
        if "withdrawn" not in history.columns:
            history["withdrawn"] = 0.0
        history["withdrawn"] = history["withdrawn"].fillna(0.0)
        if day in set(history["date"]):
            history.loc[history["date"] == day, "sleeve_value"] = sleeve
            history.loc[history["date"] == day, "withdrawn"] = withdrawn
        else:
            history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    else:
        history = pd.DataFrame([row])
    history = history.sort_values("date")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_path, index=False)
    growth = 1.0
    prev_sleeve = None
    prev_withdrawn = 0.0
    for item in history.itertuples(index=False):
        value = float(item.sleeve_value)
        taken = float(item.withdrawn)
        if prev_sleeve:
            growth *= (value + taken - prev_withdrawn) / prev_sleeve
        prev_sleeve = value
        prev_withdrawn = taken
    previous = history[history["date"] < day]
    if previous.empty:
        daily = float("nan")
    else:
        prior = previous.iloc[-1]
        daily = (sleeve + withdrawn - float(prior["withdrawn"])) / float(prior["sleeve_value"]) - 1
    return {
        "sleeve": sleeve,
        "daily_return": daily,
        "total_return": growth - 1,
        "since": str(history["date"].iloc[0]),
    }


def format_report(snapshot: dict, performance: dict, day: str) -> str:
    """Text sent to Telegram. Orders in this message have not been sent."""
    currency = snapshot["currency"]
    lines = [
        f"US sleeve {day} 08:00 SGT",
        "Excluded from this return and from orders: VOO, TSM, GLD, DBS, OCBC.",
        f"Sleeve value: {snapshot['sleeve']:,.2f} {currency}",
    ]
    parked_flow = float(snapshot.get("parked_flow") or 0.0)
    if abs(parked_flow) >= 1:
        lines.append(
            f"Cash from selling parked shares is left out: {parked_flow:,.2f} {currency}."
        )
    withdrawal = float(snapshot.get("withdrawal") or 0.0)
    if withdrawal >= 1:
        lines.append(f"Cash converted out of USD is a withdrawal: {withdrawal:,.2f} {currency}.")
    if pd.notna(performance["daily_return"]):
        lines.append(f"Since yesterday: {performance['daily_return']:+.2%}")
    lines.append(f"Since {performance['since']}: {performance['total_return']:+.2%}")
    lines.append("")
    lines.append("Target weights")
    targets = snapshot["plan"]
    targets = targets[targets["weight"] > 0].sort_values("weight", ascending=False)
    for row in targets.itertuples(index=False):
        lines.append(
            f"{row.ticker} {row.weight:.1%} -> {int(row.target_shares)} shares "
            f"(held {int(row.current_shares)})"
        )
    lines.append("")
    orders = snapshot["orders"]
    if orders.empty:
        lines.append("No orders. Holdings already match the target book.")
    else:
        lines.append("Orders waiting for approval")
        for row in orders.itertuples(index=False):
            lines.append(f"{row.action} {int(abs(row.delta_shares))} {row.tiger_symbol}")
        lines.append("")
        lines.append("Reply approve or execute to send these market orders.")
        lines.append("Reply reject to discard them.")
        lines.append(
            "Before 9:30pm SGT the approval is held. Market orders go out when the US cash session opens."
        )
    return "\n".join(lines)


def pending_path(market: str = "us") -> Path:
    _, processed = get_paths(market)
    return processed / "pending_rebalance.json"


def save_pending(snapshot: dict, day: str, market: str = "us") -> Path:
    orders = snapshot["orders"]
    payload = {
        "date": day,
        "market": market,
        "currency": snapshot["currency"],
        "status": "pending" if not orders.empty else "empty",
        "orders": orders.to_dict(orient="records") if not orders.empty else [],
    }
    path = pending_path(market)
    path.write_text(json.dumps(payload, default=str), encoding="utf-8")
    return path


def session_phase(plan_day: str, moment: datetime | None = None) -> str:
    """Where ``moment`` sits relative to the US cash session on ``plan_day``.

    The session is 09:30-16:00 America/New_York. A plan dated on an SGT morning
    belongs to the US session later that same New York date.
    """
    moment = (moment or datetime.now(ET)).astimezone(ET)
    session_day = date.fromisoformat(plan_day)
    if session_day.weekday() >= 5:
        return "closed"
    open_at = datetime.combine(session_day, clock_time(9, 30), tzinfo=ET)
    close_at = datetime.combine(session_day, clock_time(16, 0), tzinfo=ET)
    if moment < open_at:
        return "early"
    if moment < close_at:
        return "open"
    return "closed"


def _read_pending(market: str = "us") -> dict | None:
    path = pending_path(market)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_pending(payload: dict, market: str) -> None:
    pending_path(market).write_text(json.dumps(payload, default=str), encoding="utf-8")


def _order_key(row: dict) -> tuple:
    symbol = row.get("tiger_symbol") or row.get("ticker")
    quantity = row.get("quantity", row.get("delta_shares"))
    if quantity is None:
        return (row.get("action"), symbol, None)
    return (row.get("action"), symbol, abs(int(quantity)))


def remaining_orders(payload: dict) -> list[dict]:
    """Plan rows that do not already have an accepted Tiger order."""
    done = {_order_key(row) for row in payload.get("submitted") or []}
    remaining = []
    for row in payload.get("orders") or []:
        quantity = row.get("delta_shares", row.get("quantity"))
        if quantity is not None and abs(int(quantity)) < 1:
            continue
        if _order_key(row) in done:
            continue
        remaining.append(row)
    return remaining


def claim_pending(day: str, market: str = "us", statuses: tuple[str, ...] = ("pending",)) -> dict | None:
    """Mark an unsent plan as executing. A second approve finds nothing to send."""
    with _PLAN_LOCK:
        payload = _read_pending(market)
        if payload is None:
            return None
        if payload.get("date") != day or payload.get("status") not in statuses:
            return None
        if not remaining_orders(payload):
            return None
        payload["status"] = "executing"
        _write_pending(payload, market)
        return payload


def finish_pending(day: str, status: str, market: str = "us") -> None:
    with _PLAN_LOCK:
        payload = _read_pending(market)
        if payload is None or payload.get("date") != day:
            return
        payload["status"] = status
        _write_pending(payload, market)


def _api(token: str, method: str, payload: dict | None = None) -> dict:
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = None if payload is None else urllib.parse.urlencode(payload).encode()
    request = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not body.get("ok"):
        raise RuntimeError(body.get("description") or "Telegram request failed")
    return body


def send_message(token: str, chat_id: str, text: str) -> None:
    step = 3500
    for start in range(0, len(text), step):
        _api(token, "sendMessage", {"chat_id": chat_id, "text": text[start : start + step]})


def is_approval(text: str) -> bool:
    return text.strip().lower() in APPROVE


def is_rejection(text: str) -> bool:
    return text.strip().lower() in REJECT


def _plan_reply(payload: dict | None, day: str) -> str:
    if payload and payload.get("date") == day and payload.get("status") == "sent":
        return "Today's orders were already sent."
    if payload and payload.get("date") == day and payload.get("status") == "rejected":
        return "Today's plan was discarded. Nothing was sent."
    if payload and payload.get("date") == day and payload.get("status") == "executing":
        return "A send is already in progress."
    if payload and payload.get("submitted"):
        sent = ", ".join(row["tiger_symbol"] for row in payload["submitted"])
        return f"Some orders were already sent ({sent}). Nothing more was sent."
    return "No unsent plan for today. Nothing was sent."


def _queue_plan(day: str, market: str) -> str:
    with _PLAN_LOCK:
        payload = _read_pending(market)
        if payload is None or payload.get("date") != day or not payload.get("orders"):
            return _plan_reply(payload, day)
        if payload.get("status") not in {"pending", "queued", "failed"} or payload.get("submitted"):
            return _plan_reply(payload, day)
        payload["status"] = "queued"
        _write_pending(payload, market)
    return HELD_MESSAGE


def _hours_rejection(exc: BaseException) -> bool:
    return "regular trading hours" in str(exc).lower()


def _rate_limited(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "requestrateexceedlimit" in text or "too_many_requests" in text or "rate limit" in text


def _send_plan(day: str, market: str) -> str:
    """Send a plan that was approved. Caller has already checked the session is open."""
    from broker.tiger_client import place_orders, trade_client

    payload = claim_pending(day, market, statuses=("pending", "queued", "failed"))
    if payload is None:
        return _plan_reply(_read_pending(market), day)
    prior = list(payload.get("submitted") or [])
    orders = pd.DataFrame(remaining_orders(payload))
    try:
        sent = place_orders(trade_client(), orders, payload["currency"])
    except Exception as exc:
        sent = prior + list(getattr(exc, "sent", []))
        with _PLAN_LOCK:
            current = _read_pending(market) or payload
            current["submitted"] = sent
            if not sent and _hours_rejection(exc):
                current["status"] = "queued"
            elif _rate_limited(exc):
                current["status"] = "queued"
            else:
                current["status"] = "failed"
            _write_pending(current, market)
        if not prior and not getattr(exc, "sent", []) and _hours_rejection(exc):
            return HELD_MESSAGE
        detail = f"Order send failed: {exc}"
        if sent:
            done = ", ".join(f"{row['action']} {row['quantity']} {row['tiger_symbol']}" for row in sent)
            detail = f"{detail}\nAlready sent: {done}"
        return detail
    submitted = prior + sent
    with _PLAN_LOCK:
        current = _read_pending(market) or payload
        current["submitted"] = submitted
        current["status"] = "sent"
        _write_pending(current, market)
    if not sent:
        return "Plan was claimed but no whole-share order was sent."
    lines = [f"Sent {row['action']} {row['quantity']} {row['tiger_symbol']}" for row in sent]
    return "\n".join(lines)


def execute_pending(day: str | None = None, market: str = "us") -> str:
    """Send the saved plan once the US cash session is open. Hold it before that."""
    day = day or now_sgt().date().isoformat()
    phase = session_phase(day)
    if phase == "early":
        return _queue_plan(day, market)
    if phase != "open":
        return "The US cash session for this plan has ended. Nothing was sent."
    return _send_plan(day, market)


def flush_queued(market: str = "us") -> str | None:
    """Send an approval that was held for the open. No-op until that session starts."""
    payload = _read_pending(market)
    if payload is None or payload.get("status") != "queued" or not payload.get("date"):
        return None
    if session_phase(str(payload["date"])) != "open":
        return None
    return _send_plan(str(payload["date"]), market)


def publish_report(market: str = "us") -> str:
    """Pull the account, store the sleeve value, save the plan, and return the message."""
    from broker.session import prepare_rebalance

    day = now_sgt().date().isoformat()
    snapshot = prepare_rebalance(market)
    _, processed = get_paths(market)
    performance = record_sleeve(
        processed / "sleeve_history.csv",
        day,
        snapshot["sleeve"],
        float(snapshot.get("withdrawn") or 0.0),
    )
    save_pending(snapshot, day, market)
    return format_report(snapshot, performance, day)


def report(market: str = "us") -> None:
    env = load_env()
    text = publish_report(market)
    send_message(env["TELEGRAM_BOT_TOKEN"], env["TELEGRAM_CHAT_ID"], text)
    print(text)


def listen() -> None:
    """Wait for approve, execute, or reject from the configured chat only."""
    env = load_env()
    token = env["TELEGRAM_BOT_TOKEN"]
    chat_id = str(env["TELEGRAM_CHAT_ID"])
    offset = 0
    print("Listening for approve, execute, or reject.")
    while True:
        try:
            body = _api(token, "getUpdates", {"timeout": 25, "offset": offset})
        except Exception as exc:
            print(f"Telegram poll failed: {exc}")
            time.sleep(5)
            continue
        for update in body.get("result") or []:
            offset = int(update["update_id"]) + 1
            message = update.get("message") or {}
            if str(message.get("chat", {}).get("id")) != chat_id:
                continue
            text = message.get("text") or ""
            if is_rejection(text):
                finish_pending(now_sgt().date().isoformat(), "rejected")
                send_message(token, chat_id, "Discarded. Nothing was sent.")
            elif is_approval(text):
                reply = execute_pending()
                send_message(token, chat_id, reply)


def serve() -> None:
    """Send the 8:00am SGT report once a day, and keep listening for approval."""
    import threading

    def _daily() -> None:
        sent_on = ""
        existing = _read_pending()
        today = now_sgt().date().isoformat()
        if existing and existing.get("date") == today:
            sent_on = today
        while True:
            current = now_sgt()
            day = current.date().isoformat()
            if current.hour >= 8 and sent_on != day:
                try:
                    report()
                    sent_on = day
                except Exception as exc:
                    print(f"Morning report failed: {exc}")
                    time.sleep(60)
                    continue
            try:
                queued = flush_queued()
            except Exception as exc:
                print(f"Queued send failed: {exc}")
                queued = None
            if queued:
                try:
                    env = load_env()
                    send_message(env["TELEGRAM_BOT_TOKEN"], env["TELEGRAM_CHAT_ID"], queued)
                    print(queued)
                except Exception as exc:
                    print(f"Queued send notice failed: {exc}")
            time.sleep(30)

    threading.Thread(target=_daily, daemon=True).start()
    listen()


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "report"
    if command == "report":
        report()
    elif command == "listen":
        listen()
    elif command == "serve":
        serve()
    elif command == "execute":
        print(execute_pending())
    elif command == "whoami":
        env = load_env()
        body = _api(env["TELEGRAM_BOT_TOKEN"], "getUpdates", {"timeout": 1})
        chats = []
        for update in body.get("result") or []:
            chat = (update.get("message") or {}).get("chat") or {}
            if chat.get("id") is not None:
                chats.append(str(chat["id"]))
        print("Chat ids that have messaged this bot:")
        print("\n".join(chats) or "None yet. Open the bot in Telegram and send hi, then run this again.")
    else:
        raise SystemExit("Use report, listen, serve, execute, or whoami.")


if __name__ == "__main__":
    main()
