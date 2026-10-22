"""Polymarket US (CFTC-regulated) Retail API client: key-file loading, Ed25519 request signing, orders.

Docs used (docs.polymarket.us): Private-key auth = Ed25519 over f"{timestamp_ms}{METHOD}{path}" (no body, no query string),
headers X-PM-Access-Key / X-PM-Timestamp / X-PM-Signature; timestamp must be within 30 s of server time.
Order `price.value` is ALWAYS the long (YES) price - buying NO at X is sent as BUY_SHORT with price = 1 - X.

SECURITY: the secret is only held in memory for signing; it is never logged, printed or placed in an exception message.
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional

import requests
from cryptography.hazmat.primitives.asymmetric import ed25519

API_BASE = "https://api.polymarket.us"


def api_base() -> str:
    """Production host. Test hook: POLYUS_API_BASE may redirect to a LOOPBACK mock server only (never an external host)."""
    o = os.environ.get("POLYUS_API_BASE", "")
    return o if o.startswith(("http://127.0.0.1", "http://localhost")) else API_BASE


GATEWAY_BASE = "https://gateway.polymarket.us"

UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


class CredentialError(Exception):
    pass


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


class BrokerError(Exception):
    """Order state could not be determined - callers must stop trading rather than guess."""


# ── credentials ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Credentials:
    key_id: str
    secret_b64: str

    def masked(self) -> str:
        return f"{self.key_id[:8]}…"

    def __repr__(self) -> str:                       # never leak the secret via repr/logging
        return f"Credentials(key_id={self.masked()}, secret=<hidden>)"


def candidate_dirs() -> list[Path]:
    home = Path.home()
    out = [home / "Desktop", home / "OneDrive" / "Desktop", home / "OneDrive" / "Documents" / "Desktop",
           home / "Downloads", home / "Documents"]
    try:                                              # real Windows Desktop (handles OneDrive redirection)
        import ctypes
        from ctypes import wintypes
        buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
        if ctypes.windll.shell32.SHGetFolderPathW(None, 0x0000, None, 0, buf) == 0:      # CSIDL_DESKTOP
            out.insert(0, Path(buf.value))
    except Exception:
        pass
    seen, uniq = set(), []
    for d in out:
        if str(d) not in seen:
            seen.add(str(d))
            uniq.append(d)
    return uniq


def find_key_file() -> Optional[Path]:
    """Newest file named like 'Polymarket Key*.txt|json' on the Desktop, then Downloads/Documents."""
    for d in candidate_dirs():
        if not d.is_dir():
            continue
        hits = [p for p in d.glob("*") if p.is_file() and re.search(r"polymarket", p.name, re.I)
                and p.suffix.lower() in (".txt", ".json", ".key", ".env", "")]
        if hits:
            return max(hits, key=lambda p: p.stat().st_mtime)
    return None


def parse_credentials(text: str) -> Credentials:
    """Accepts 'Key ID: …' / 'Secret Key: …' lines, KEY=VALUE lines, or a JSON object."""
    text = text.lstrip("﻿").strip()
    key_id = secret = None
    if text.startswith("{"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise CredentialError("key file looks like JSON but is not valid JSON") from None
        lower = {str(k).lower().replace("_", "").replace(" ", ""): v for k, v in obj.items()}
        key_id = lower.get("keyid") or lower.get("accesskey") or lower.get("apikey")
        secret = lower.get("secretkey") or lower.get("secret") or lower.get("privatekey")
    else:
        for line in text.splitlines():
            m = re.match(r"^\s*([A-Za-z _\-]+?)\s*[:=]\s*[\"']?([^\"'\s]+)[\"']?\s*$", line)
            if not m:
                continue
            label, value = m.group(1).lower().replace("_", "").replace(" ", "").replace("-", ""), m.group(2)
            if label in ("keyid", "polymarketkeyid", "accesskey", "apikey", "id"):
                key_id = value
            elif label in ("secretkey", "polymarketsecretkey", "secret", "privatekey"):
                secret = value
    if not key_id or not secret:
        raise CredentialError("could not find both 'Key ID' and 'Secret Key' in the key file")
    if not UUID_RE.match(key_id):
        raise CredentialError("Key ID is not a UUID - is this the right file?")
    try:
        raw = base64.b64decode(secret, validate=True)
    except Exception:
        raise CredentialError("Secret Key is not valid base64") from None
    if len(raw) < 32:
        raise CredentialError("Secret Key decodes to fewer than 32 bytes")
    return Credentials(key_id=key_id, secret_b64=secret)


def load_credentials(path: str | Path) -> Credentials:
    try:
        return parse_credentials(Path(path).read_text(encoding="utf-8-sig", errors="strict"))
    except OSError as e:
        raise CredentialError(f"cannot read key file: {e.strerror}") from None
    except UnicodeDecodeError:
        raise CredentialError("key file is not UTF-8 text") from None


# ── signing ───────────────────────────────────────────────────────────────────
class Signer:
    def __init__(self, creds: Credentials):
        self.key_id = creds.key_id
        self._sk = ed25519.Ed25519PrivateKey.from_private_bytes(base64.b64decode(creds.secret_b64)[:32])

    def public_key(self):
        return self._sk.public_key()

    def headers(self, method: str, path: str, ts_ms: Optional[int] = None) -> dict:
        ts = str(ts_ms if ts_ms is not None else int(time.time() * 1000))
        sig = base64.b64encode(self._sk.sign(f"{ts}{method.upper()}{path}".encode())).decode()
        return {"X-PM-Access-Key": self.key_id, "X-PM-Timestamp": ts, "X-PM-Signature": sig,
                "Content-Type": "application/json"}


# ── helpers ───────────────────────────────────────────────────────────────────
def window_slug(open_ts: int) -> str:
    """15-minute window starting at ``open_ts`` (unix s, UTC) -> Polymarket US market slug."""
    dt = datetime.fromtimestamp(open_ts, timezone.utc)
    return f"cpc-btc-updown-15m-{dt:%Y-%m-%d-%H%M}z"


def amount(x) -> Optional[float]:
    """Amount object {'value': '0.55', 'currency': 'USD'} | str | number -> float."""
    if x is None:
        return None
    if isinstance(x, dict):
        x = x.get("value")
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


SLUG_PREFIX = "cpc-btc-updown-15m-"


def parse_trade_activities(acts: list) -> list:
    """Raw activities -> [{'slug', 'price', 'qty', 'time', 'id', 'realized_pnl'}] for this strategy's market family only (BTC 15-minute up/down)."""
    out = []
    for a in acts:
        t = a.get("trade") or {}
        slug = str(t.get("marketSlug", ""))
        if not slug.startswith(SLUG_PREFIX):
            continue
        qty = t.get("qtyDecimal", t.get("qty"))
        try:
            qty = float(qty)
        except (TypeError, ValueError):
            qty = float("nan")
        out.append(dict(slug=slug, price=amount(t.get("price")), qty=qty, time=str(t.get("createTime", "")), id=str(t.get("id", "")), realized_pnl=amount(t.get("realizedPnl"))))
    return out


def build_buy_order(slug: str, side: str, contracts: int, max_cost: float, sync: bool = True) -> dict:
    """IOC limit BUY of the YES or NO side, never paying more than ``max_cost`` per contract (that side's price).
    price.value is the LONG (YES) price: YES -> floor to the cent; NO -> 1 - max_cost, rounded UP so NO cost <= max_cost."""
    if side not in ("YES", "NO"):
        raise ValueError(side)
    if side == "YES":
        long_price = math.floor(round(max_cost * 100, 6)) / 100
        intent = "ORDER_INTENT_BUY_LONG"
    else:
        long_price = math.ceil(round((1.0 - max_cost) * 100, 6)) / 100
        intent = "ORDER_INTENT_BUY_SHORT"
    long_price = min(max(long_price, 0.01), 0.99)
    body = {
        "marketSlug": slug,
        "type": "ORDER_TYPE_LIMIT",
        "price": {"value": f"{long_price:.2f}", "currency": "USD"},
        "quantity": int(contracts),
        "tif": "TIME_IN_FORCE_IMMEDIATE_OR_CANCEL",
        "intent": intent,
        "manualOrderIndicator": "MANUAL_ORDER_INDICATOR_AUTOMATIC",
    }
    if sync:
        body["synchronousExecution"] = True
        body["maxBlockTime"] = "5"
    return body


def side_price_from_long(side: str, long_price: float) -> float:
    return long_price if side == "YES" else 1.0 - long_price


TERMINAL_STATES = {"ORDER_STATE_FILLED", "ORDER_STATE_CANCELED", "ORDER_STATE_REJECTED", "ORDER_STATE_EXPIRED"}


@dataclass
class Fill:
    qty: int
    price: float                 # per-contract price of the side we bought (YES price or NO price)
    fee: float
    order_id: str
    state: str = ""
    note: str = ""


def parse_order_fill(order: dict, side: str, max_cost: float, contracts: int) -> Optional[Fill]:
    """Order object (GET /v1/order/{id}) -> Fill (or None if nothing filled). Raises BrokerError if state is not terminal."""
    state = order.get("state", "")
    filled = float(order.get("cumQuantity") or 0.0)
    if state not in TERMINAL_STATES and not (state == "ORDER_STATE_PARTIALLY_FILLED" and float(order.get("leavesQuantity") or 0) == 0):
        raise BrokerError(f"order not in a terminal state ({state or 'unknown'})")
    if filled <= 0:
        return None
    avg = amount(order.get("avgPx"))
    if avg is None:
        raise BrokerError("filled order has no average price")
    # avgPx follows the long-price convention for short orders; accept whichever interpretation respects our limit
    cands = [avg] if side == "YES" else [1.0 - avg, avg]
    price = next((c for c in cands if 0.005 <= c <= max_cost + 1e-6), None)
    if price is None:
        raise BrokerError(f"fill price {avg} violates the limit {max_cost:.2f} under both price conventions")
    fee = amount(order.get("commissionNotionalTotalCollected"))
    return Fill(qty=int(round(filled)), price=float(price), fee=abs(fee) if fee is not None else float("nan"),
                order_id=str(order.get("id", "")), state=state, note=f"exchange avgPx={avg:g}")          # the raw number is kept in the ledger so fill prices can be audited against the account


# ── client ────────────────────────────────────────────────────────────────────
class PolyUSClient:
    def __init__(self, creds: Credentials, base: Optional[str] = None, timeout: float = 10.0, session: Optional[requests.Session] = None):
        self.signer = Signer(creds)
        self.base = (base or api_base()).rstrip("/")
        self.timeout = timeout
        self.s = session or requests.Session()
        self.skew_s: Optional[float] = None            # server clock - local clock (seconds), from the Date header

    def _req(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        data = json.dumps(body, separators=(",", ":")) if body is not None else None
        t0 = time.time()
        try:
            r = self.s.request(method, self.base + path, headers=self.signer.headers(method, path.split("?")[0]), data=data, timeout=self.timeout)       # the signature covers the path only, never the query
        except requests.RequestException as e:
            raise ApiError(0, f"network error ({type(e).__name__})") from None
        d = r.headers.get("Date")
        if d:
            try:
                self.skew_s = parsedate_to_datetime(d).timestamp() - (t0 + time.time()) / 2
            except Exception:
                pass
        if r.status_code >= 400:
            msg = r.text[:300].replace("\n", " ")
            if r.status_code == 401:
                msg = "unauthorized - check the key file and that this PC's clock is within 30 s of real time"
            raise ApiError(r.status_code, msg)
        return r.json() if r.content else {}

    # account
    def balances(self) -> dict:
        res = self._req("GET", "/v1/account/balances")
        for b in res.get("balances", []):
            if str(b.get("currency", "USD")).upper() == "USD":
                return {"balance": float(b.get("currentBalance") or 0), "buying_power": float(b.get("buyingPower") or 0),
                        "open_orders": float(b.get("openOrders") or 0), "unsettled": float(b.get("unsettledFunds") or 0)}
        return {"balance": 0.0, "buying_power": 0.0, "open_orders": 0.0, "unsettled": 0.0}

    def positions(self) -> dict:
        return self._req("GET", "/v1/portfolio/positions").get("positions", {})

    def open_orders(self) -> list:
        return self._req("GET", "/v1/orders/open").get("orders", [])

    def trade_activities(self, max_pages: int = 5) -> list:
        """The account's recent trades, newest first (GET /v1/portfolio/activities?types=ACTIVITY_TYPE_TRADE, paginated by cursor).  Read-only.  Built from the public docs - not yet run against a real account."""
        from urllib.parse import quote
        out, cur = [], None
        for _ in range(max_pages):
            d = self._req("GET", "/v1/portfolio/activities?types=ACTIVITY_TYPE_TRADE" + (f"&cursor={quote(str(cur), safe='')}" if cur else ""))
            out += [a for a in d.get("activities", []) if isinstance(a, dict) and isinstance(a.get("trade"), dict)]
            cur = d.get("nextCursor")
            if d.get("eof") or not cur:
                break
        return out

    # trading
    def preview(self, order: dict) -> dict:
        return self._req("POST", "/v1/order/preview", {"request": order})

    def create_order(self, order: dict) -> dict:
        return self._req("POST", "/v1/orders", order)

    def get_order(self, order_id: str) -> dict:
        return self._req("GET", f"/v1/order/{order_id}").get("order", {})

    def cancel_all(self, slugs: Optional[list] = None) -> list:
        return self._req("POST", "/v1/orders/open/cancel", {"slugs": slugs or []}).get("canceledOrderIds", [])

    def book(self, slug: str) -> dict:
        return self._req("GET", f"/v1/markets/{slug}/book").get("marketData", {})


class PolyBroker:
    """Real orders. Every buy is an IOC limit order; the fill is read back from GET /v1/order/{id}."""
    mode = "live"

    def __init__(self, client: PolyUSClient, poll_s: float = 8.0):
        self.c = client
        self.poll_s = poll_s

    def buying_power(self) -> Optional[float]:
        return self.c.balances()["buying_power"]

    def buy(self, slug: str, side: str, contracts: int, max_cost: float, quoted_cost: float) -> Optional[Fill]:
        body = build_buy_order(slug, side, contracts, max_cost)
        try:
            resp = self.c.create_order(body)
        except ApiError as e:
            if 400 <= e.status < 500 and e.status not in (401, 408, 429):
                raise BrokerError(f"order rejected by API: {e.message}") from None
            raise BrokerError(f"order submission failed ({e.status}); order state unknown") from None
        oid = str(resp.get("id", ""))
        if not oid:
            raise BrokerError("order accepted but no order id returned; state unknown")
        deadline = time.time() + self.poll_s
        last_err = ""
        while time.time() < deadline:
            try:
                order = self.c.get_order(oid)
                fill = parse_order_fill(order, side, max_cost, contracts)
                if fill is not None and fill.fee != fill.fee:             # NaN fee -> compute from the formula
                    from .fees import fee_total
                    fill.fee = fee_total("Polymarket US", fill.qty, fill.price)
                return fill
            except BrokerError as e:
                last_err = str(e)
            except ApiError as e:
                last_err = str(e)
            time.sleep(0.4)
        try:                                                              # still unresolved: kill any remainder, then give up loudly
            self.c.cancel_all([slug])
        except Exception:
            pass
        raise BrokerError(f"order {oid} unresolved after {self.poll_s:.0f}s ({last_err}); remainder cancelled - check the account")

    def cancel_all(self) -> None:
        self.c.cancel_all([])
