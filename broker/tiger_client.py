"""Tiger Open API account, positions, and order placement."""

from __future__ import annotations

import os
import time
from pathlib import Path

from config import ROOT

PROPS_NAME = "tiger_openapi_config.properties"

_MARKET = {"us": ("US", "USD"), "sg": ("SG", "SGD")}


def _import_tiger():
    try:
        from tigeropen.common.consts import Currency, Market, SecurityType
        from tigeropen.common.util.contract_utils import stock_contract
        from tigeropen.common.util.order_utils import market_order
        from tigeropen.tiger_open_config import TigerOpenClientConfig
        from tigeropen.trade.trade_client import TradeClient
    except ImportError as exc:
        raise SystemExit(
            "Tiger SDK is not installed. Run: pip install tigeropen"
        ) from exc
    return {
        "Currency": Currency,
        "Market": Market,
        "SecurityType": SecurityType,
        "stock_contract": stock_contract,
        "market_order": market_order,
        "TigerOpenClientConfig": TigerOpenClientConfig,
        "TradeClient": TradeClient,
    }


def props_file() -> Path | None:
    """Directory or file that holds Tiger credentials, if one is configured."""
    override = os.environ.get("TIGER_PROPS_PATH")
    if override:
        return Path(override).expanduser()
    default = ROOT / PROPS_NAME
    if default.is_file():
        return default
    return None


def load_client_config():
    """Build a Tiger client config from a properties file or environment variables."""
    sdk = _import_tiger()
    TigerOpenClientConfig = sdk["TigerOpenClientConfig"]
    located = props_file()
    if located is not None:
        props_dir = located if located.is_dir() else located.parent
        config_file = props_dir / PROPS_NAME
        if not config_file.is_file():
            raise SystemExit(f"Tiger config not found: {config_file}")
        return TigerOpenClientConfig(props_path=str(props_dir))

    tiger_id = os.environ.get("TIGER_ID")
    account = os.environ.get("TIGER_ACCOUNT")
    key_path = os.environ.get("TIGER_PRIVATE_KEY_PATH")
    if not (tiger_id and account and key_path):
        raise SystemExit(
            "Tiger credentials are missing. Copy "
            "broker/tiger_openapi_config.properties.example to "
            "tiger_openapi_config.properties in the project folder and fill it in, "
            "or set TIGER_ID, TIGER_ACCOUNT, and TIGER_PRIVATE_KEY_PATH."
        )
    from tigeropen.common.util.signature_utils import read_private_key

    config = TigerOpenClientConfig()
    config.tiger_id = tiger_id
    config.account = account
    config.private_key = read_private_key(key_path)
    license_name = os.environ.get("TIGER_LICENSE")
    if license_name:
        config.license = license_name
    return config


def trade_client():
    sdk = _import_tiger()
    return sdk["TradeClient"](load_client_config())


def _market_currency(market: str):
    sdk = _import_tiger()
    code, currency_name = _MARKET[market]
    market_enum = getattr(sdk["Market"], code, code)
    currency_enum = getattr(sdk["Currency"], currency_name, currency_name)
    return market_enum, currency_enum, currency_name


def _account_record(assets):
    if isinstance(assets, list):
        if not assets:
            raise RuntimeError("Tiger returned no account assets")
        return assets[0]
    if assets is None:
        raise RuntimeError("Tiger returned no account assets")
    return assets


def net_liquidation(client, market: str) -> float:
    """Securities-segment net liquidation in the market currency."""
    _, _, currency_name = _market_currency(market)
    assets = _account_record(client.get_prime_assets(base_currency=currency_name))
    segments = getattr(assets, "segments", None) or {}
    segment = segments.get("S") or (next(iter(segments.values())) if segments else None)
    if segment is None:
        raise RuntimeError("Tiger account has no securities segment")
    nav = float(segment.net_liquidation)
    if nav <= 0:
        raise RuntimeError(f"Tiger net liquidation is {nav}")
    return nav


def _shares(position) -> float:
    qty = getattr(position, "position_qty", None)
    if qty is not None:
        return float(qty)
    return float(position.quantity)


def _currency_cash(segment, currency_name: str) -> float:
    assets = getattr(segment, "currency_assets", None) or {}
    bucket = assets.get(currency_name)
    if bucket is None:
        return 0.0
    cash = getattr(bucket, "cash_balance", None)
    if cash is None:
        return 0.0
    return float(cash)


def market_book(client, market: str) -> dict:
    """Cash plus stock market value for one market, with live share counts and prices."""
    from broker.plan import tiger_to_yahoo

    sdk = _import_tiger()
    market_enum, _, currency_name = _market_currency(market)
    assets = _account_record(client.get_prime_assets(base_currency=currency_name))
    segments = getattr(assets, "segments", None) or {}
    segment = segments.get("S") or (next(iter(segments.values())) if segments else None)
    if segment is None:
        raise RuntimeError("Tiger account has no securities segment")
    cash = _currency_cash(segment, currency_name)

    positions = client.get_positions(
        sec_type=sdk["SecurityType"].STK,
        market=market_enum,
    ) or []
    held: dict[str, float] = {}
    prices: dict[str, float] = {}
    stock_value = 0.0
    for position in positions:
        contract = position.contract
        symbol = getattr(contract, "symbol", None)
        if not symbol:
            continue
        shares = _shares(position)
        if abs(shares) < 1e-9:
            continue
        ticker = tiger_to_yahoo(str(symbol), market)
        held[ticker] = shares
        market_value = getattr(position, "market_value", None)
        if market_value is not None:
            stock_value += float(market_value)
        price = getattr(position, "market_price", None)
        if price is not None and float(price) > 0:
            prices[ticker] = float(price)
    return {
        "currency": currency_name,
        "cash": cash,
        "stock_value": stock_value,
        "asset_value": cash + stock_value,
        "positions": held,
        "prices": prices,
    }


class OrderSendError(Exception):
    """Tiger rejected an order after zero or more earlier orders were accepted."""

    def __init__(self, message: str, sent: list[dict]):
        super().__init__(message)
        self.sent = sent


def _order_rate_limited(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "requestrateexceedlimit" in text or "too_many_requests" in text or "rate limit" in text


def place_orders(client, orders, currency: str) -> list[dict]:
    """Submit market orders. ``orders`` rows need tiger_symbol, action, delta_shares.

    Tiger rejects a burst of orders, so each order waits and a rate-limit
    response is retried instead of abandoning the rest of the list.
    """
    sdk = _import_tiger()
    account = getattr(client, "_account", None) or load_client_config().account
    sent = []
    rows = list(orders.itertuples(index=False))
    for index, row in enumerate(rows):
        quantity = abs(int(row.delta_shares))
        if quantity < 1:
            continue
        contract = sdk["stock_contract"](row.tiger_symbol, currency)
        order = sdk["market_order"](account, contract, row.action, quantity)
        order_id = None
        for attempt in range(6):
            try:
                order_id = client.place_order(order)
                break
            except Exception as exc:
                if not _order_rate_limited(exc) or attempt == 5:
                    raise OrderSendError(str(exc), sent) from exc
                wait = 3 + attempt * 2
                print(f"Tiger rate limit on {row.action} {quantity} {row.tiger_symbol}. Waiting {wait}s.")
                time.sleep(wait)
        sent.append(
            {
                "tiger_symbol": row.tiger_symbol,
                "action": row.action,
                "quantity": quantity,
                "order_id": order_id,
            }
        )
        print(f"Sent {row.action} {quantity} {row.tiger_symbol} (order {order_id})")
        if index < len(rows) - 1:
            time.sleep(1.5)
    return sent
