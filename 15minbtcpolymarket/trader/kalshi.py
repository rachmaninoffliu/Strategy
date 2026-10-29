"""Kalshi: the ONLY venue of this app - signal quotes, order-book depth, results, candle volume (public) and the signed trading API (balance, V2 orders, fills).

Trading API (docs.kalshi.com): every private request carries KALSHI-ACCESS-KEY, KALSHI-ACCESS-TIMESTAMP (ms) and KALSHI-ACCESS-SIGNATURE = base64( RSA-PSS-SHA256 | Ed25519 of
f"{timestamp}{METHOD}{path}" ), path = the full URL path from the API root ('/trade-api/v2/...'), NO query string.  Orders use the V2 single-book model (POST /portfolio/events/orders):
side 'bid' = buy YES at `price`; side 'ask' = sell YES at `price`, which is how a NO is bought (buying NO at q = an ask at YES-price 1 - q).  IOC orders return their final fill count at once.

SECURITY: the private key is only held in memory for signing; it is never logged, printed or put in an exception message.
"""
from __future__ import annotations

import base64
import json
import math
import os
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

from .broker import BrokerError, Fill, InsufficientFunds, NotPlaced

SERIES = "KXBTC15M"


@dataclass(frozen=True)
class Env:
    name: str
    label: str
    base: str


ENVS = {
    "production": Env("production", "Production — real money", "https://external-api.kalshi.com/trade-api/v2"),
    "demo": Env("demo", "Demo — play money", "https://external-api.demo.kalshi.co/trade-api/v2"),
}
BASE = ENVS["production"].base                    # default host for public reads (tests patch it)


def base_url(env: str = "production") -> str:
    """REST base of an environment.  Test hook: KALSHI_API_BASE may redirect to a LOOPBACK mock server only (never an external host)."""
    o = os.environ.get("KALSHI_API_BASE", "")
    if o.startswith(("http://127.0.0.1", "http://localhost")):
        return o.rstrip("/")
    return ENVS[env].base


class CredentialError(Exception):
    pass


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


class RateLimiter:
    """Token bucket shared by every request this process sends to one Kalshi API area: it spaces requests out instead of letting a retry loop, a refresh or a burst of calls push the app over Kalshi's limit
    (HTTP 429).  ``acquire`` waits at most ``max_wait`` seconds and then lets the request go anyway - the limiter must never hold up an order for long."""

    def __init__(self, rate: float, burst: int):
        self.rate, self.burst = float(rate), float(burst)
        self._tokens, self._t, self._lock = float(burst), time.monotonic(), threading.Lock()

    def acquire(self, max_wait: float = 1.0) -> float:
        start = time.monotonic()
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.burst, self._tokens + (now - self._t) * self.rate)
                self._t = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return now - start
                need = (1.0 - self._tokens) / self.rate
            if time.monotonic() - start + need > max_wait:
                return time.monotonic() - start
            time.sleep(need)


PUBLIC_LIMITER = RateLimiter(4.0, 4)         # market data (order book, quotes, results): Kalshi's unauthenticated limit bites around 6 requests/s, shared with anything else on this IP
PRIVATE_LIMITER = RateLimiter(8.0, 6)        # signed account / order requests
RETRY_STATUS = (429, 502, 503, 504)          # transient: worth another try for a read


def _loopback(url: str) -> bool:
    return url.startswith(("http://127.0.0.1", "http://localhost"))          # the test mock: no limiter, no waiting


def _retry_delay(resp, attempt: int, cap: float = 3.0) -> float:
    """How long to wait before re-sending after a transient error: Kalshi's Retry-After when it sends one, else 0.6 s, 1.2 s ... plus a little jitter."""
    try:
        ra = float(resp.headers.get("Retry-After", ""))
        if ra >= 0:
            return min(cap, ra) + random.uniform(0.0, 0.1)
    except (TypeError, ValueError):
        pass
    return min(cap, 0.6 * (2 ** attempt)) + random.uniform(0.0, 0.2)


def cent_floor(x: float) -> float:
    return math.floor(x * 100 + 1e-9) / 100


def cent_ceil(x: float) -> float:
    return math.ceil(x * 100 - 1e-9) / 100


# ── order book ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Book:
    """Kalshi books list BIDS only: 'yes' bids and 'no' bids.  A NO bid at p is a YES ask at 1 - p."""
    yes_bids: tuple          # ((price, contracts), ...)
    no_bids: tuple

    @staticmethod
    def from_response(d: Optional[dict]) -> Optional["Book"]:
        """Accepts the dollar format ('orderbook_fp': yes_dollars / no_dollars, strings) and the older cent format ('orderbook': yes / no, integers)."""
        ob = (d or {}).get("orderbook_fp") or (d or {}).get("orderbook")
        if not isinstance(ob, dict):
            return None
        if "yes_dollars" in ob or "no_dollars" in ob:
            ys, ns, scale = ob.get("yes_dollars") or [], ob.get("no_dollars") or [], 1.0
        else:
            ys, ns, scale = ob.get("yes") or [], ob.get("no") or [], 0.01
        try:
            yes = tuple(sorted(((round(float(p) * scale, 4), float(q)) for p, q in ys if float(q) > 0), reverse=True))
            no = tuple(sorted(((round(float(p) * scale, 4), float(q)) for p, q in ns if float(q) > 0), reverse=True))
        except (TypeError, ValueError):
            return None
        return Book(yes, no)

    def top(self) -> tuple[float, float]:
        """(best YES bid, YES ask); an empty side gives 0.0 / 1.0, which the strategy rejects."""
        return (self.yes_bids[0][0] if self.yes_bids else 0.0, round(1.0 - self.no_bids[0][0], 4) if self.no_bids else 1.0)

    def asks(self, side: str) -> list:
        """Cost levels (price of the SIDE, contracts) for buying ``side``, cheapest first."""
        src = self.no_bids if side.upper() == "YES" else self.yes_bids
        return sorted(((round(1.0 - p, 4), q) for p, q in src), key=lambda x: x[0])

    def walk(self, side: str, contracts: float, max_price: float) -> tuple[float, Optional[float]]:
        """(contracts fillable at or below ``max_price``, their average price) if ``contracts`` were bought now."""
        left, cost, filled = float(contracts), 0.0, 0.0
        for px, q in self.asks(side):
            if px > max_price + 1e-9 or left <= 1e-9:
                break
            take = min(left, q)
            cost += take * px
            filled += take
            left -= take
        return filled, (cost / filled if filled else None)


# ── public market data ────────────────────────────────────────────────────────
class KalshiQuotes:
    def __init__(self, session: Optional[requests.Session] = None, timeout: float = 6.0, base: Optional[str] = None):
        self.s = session or requests.Session()
        self.timeout = timeout
        self.base = base                              # None = module BASE (production)
        self._tickers: dict[int, str] = {}
        self._miss: dict[int, float] = {}             # window -> do not look it up again before this (monotonic) time

    def _url(self, path: str) -> str:
        return (self.base or BASE) + path

    def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        url = self._url(path)
        for i in range(3):
            if not _loopback(url):
                PUBLIC_LIMITER.acquire(1.0)
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
            except requests.RequestException:
                time.sleep(0.3)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(_retry_delay(r, i, cap=1.5))
                continue
            return None
        return None

    def window_ticker(self, open_ts: int) -> Optional[str]:
        if open_ts in self._tickers:
            return self._tickers[open_ts]
        if time.monotonic() < self._miss.get(open_ts, 0.0):
            return None                               # looked a moment ago and it was not there: do not ask again at once
        d = self._get("/markets", {"series_ticker": SERIES, "status": "open", "limit": 6})
        stamp = datetime.fromtimestamp(open_ts, timezone.utc).strftime("%Y-%m-%dT%H:%M")
        for m in (d or {}).get("markets", []):
            if str(m.get("open_time", "")).startswith(stamp):
                self._tickers[open_ts] = m["ticker"]
                return m["ticker"]
        self._miss = {k: v for k, v in self._miss.items() if v > time.monotonic()}
        self._miss[open_ts] = time.monotonic() + 1.0
        return None

    @staticmethod
    def parse_orderbook(d: Optional[dict]) -> Optional[tuple[float, float]]:
        b = Book.from_response(d)
        return b.top() if b is not None else None

    def book(self, ticker: str) -> Optional[Book]:
        """The live order book.  One request serves both the signal quote and the depth the order is checked against."""
        return Book.from_response(self._get(f"/markets/{ticker}/orderbook"))

    def market_info(self, ticker: str) -> Optional[dict]:
        """The market object (status, price ranges, ...)."""
        return (self._get(f"/markets/{ticker}") or {}).get("market")

    def exchange_index(self) -> Optional[int]:
        """The exchange shard the BTC 15-minute series trades on (Kalshi splits its markets over several matching engines; crypto is shard 2).  Collateral is checked inside that shard, so an API order is only
        funded by the cash held ON it.  None when Kalshi does not say."""
        if getattr(self, "_shard", "unset") == "unset":
            v = ((self._get(f"/series/{SERIES}") or {}).get("series") or {}).get("exchange_index")
            self._shard = int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
        return self._shard

    def shard_names(self) -> dict:
        """{exchange_index: description} from the public exchange status (best effort)."""
        out = {}
        for it in (self._get("/exchange/status") or {}).get("exchange_index_statuses") or []:
            try:
                out[int(it["exchange_index"])] = str(it.get("description", ""))
            except (KeyError, TypeError, ValueError):
                pass
        return out

    def quote(self, ticker: str) -> Optional[tuple[float, float]]:
        """(YES bid, YES ask) in dollars; (0, 1) style empty books are returned as-is for the strategy to reject.
        Read from the ORDER BOOK: the quote fields of the market object (GET /markets/{ticker}) update in steps and were seen up to ~12 s behind the book (4-5c in a fast move).
        The market object is only the fallback."""
        b = self.book(ticker)
        if b is not None:
            return b.top()
        return self.market_quote(ticker)

    def market_quote(self, ticker: str) -> Optional[tuple[float, float]]:
        """The (lagging) quote fields of the market object - only the fallback when the order book could not be read."""
        d = self._get(f"/markets/{ticker}")
        m = (d or {}).get("market")
        if not m:
            return None
        try:
            return float(m["yes_bid_dollars"]), float(m["yes_ask_dollars"])
        except (KeyError, TypeError, ValueError):
            return None

    def minute_volume(self, ticker: str, open_ts: int, minute: int) -> Optional[float]:
        """Contracts traded in the one-minute candle that ENDED at ``minute`` of the window (the last completed minute at the decision instant), or None if unavailable.
        Same definition as the historical candle volume the backtest uses (candle 'minute' = its end offset)."""
        end = int(open_ts) + 60 * int(minute)
        url = self._url(f"/series/{SERIES}/markets/{ticker}/candlesticks")
        if not _loopback(url):
            PUBLIC_LIMITER.acquire(0.3)
        try:                                                  # one short attempt: this sits on the order path, so it must not hold the order back with retries
            r = self.s.get(url, params={"start_ts": end - 60, "end_ts": end, "period_interval": 1}, timeout=min(self.timeout, 2.5))
            d = r.json() if r.status_code == 200 else None
        except (requests.RequestException, ValueError):
            return None
        for c in (d or {}).get("candlesticks", []):
            if int(c.get("end_period_ts", -1)) == end:
                try:
                    return float(c["volume_fp"])
                except (KeyError, TypeError, ValueError):
                    return None
        return None

    def result(self, ticker: str) -> Optional[str]:
        d = self._get(f"/markets/{ticker}")
        r = ((d or {}).get("market") or {}).get("result")
        return r if r in ("yes", "no") else None


# ── credentials and signing ───────────────────────────────────────────────────
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
PEM_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)


@dataclass(frozen=True)
class Credentials:
    key_id: str
    pem: str

    def masked(self) -> str:
        return f"{self.key_id[:8]}…"

    def __repr__(self) -> str:                       # never leak the key via repr/logging
        return f"Credentials(key_id={self.masked()}, key=<hidden>)"


def load_credentials(path: str | Path, key_id: str = "") -> Credentials:
    """The PEM private key file Kalshi gives you when you create an API key, plus the Key ID (typed in, or a 'Key ID: <uuid>' line in the same file)."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig", errors="strict")
    except OSError as e:
        raise CredentialError(f"cannot read key file: {e.strerror}") from None
    except UnicodeDecodeError:
        raise CredentialError("key file is not UTF-8 text") from None
    m = PEM_RE.search(text)
    if not m:
        raise CredentialError("no PEM private key ('-----BEGIN ... PRIVATE KEY-----') found in the key file")
    pem = m.group(0)
    kid = key_id.strip()
    if not kid:
        for line in text.replace(pem, "").splitlines():
            mm = re.match(r"^\s*(?:api[\s_-]*)?key[\s_-]*id\s*[:=]\s*[\"']?([0-9A-Fa-f-]{36})[\"']?\s*$", line, re.I)
            if mm:
                kid = mm.group(1)
    if not kid:
        raise CredentialError("enter the API Key ID (the UUID shown next to the key on Kalshi's API keys page)")
    if not UUID_RE.match(kid):
        raise CredentialError("the Key ID is not a UUID - copy it from Kalshi's API keys page")
    try:
        k = serialization.load_pem_private_key(pem.encode(), password=None)
    except Exception:
        raise CredentialError("the private key could not be read (is it password-protected or truncated?)") from None
    if not isinstance(k, (rsa.RSAPrivateKey, ed25519.Ed25519PrivateKey)):
        raise CredentialError("unsupported key type (Kalshi API keys are RSA or Ed25519)")
    return Credentials(key_id=kid, pem=pem)


class Signer:
    def __init__(self, creds: Credentials):
        self.key_id = creds.key_id
        self._k = serialization.load_pem_private_key(creds.pem.encode(), password=None)

    def public_key(self):
        return self._k.public_key()

    def headers(self, method: str, path: str, ts_ms: Optional[int] = None) -> dict:
        """``path`` = full URL path from the API root, without the query string."""
        ts = str(ts_ms if ts_ms is not None else int(time.time() * 1000))
        msg = f"{ts}{method.upper()}{path}".encode()
        if isinstance(self._k, rsa.RSAPrivateKey):
            sig = self._k.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
        else:
            sig = self._k.sign(msg)
        return {"KALSHI-ACCESS-KEY": self.key_id, "KALSHI-ACCESS-TIMESTAMP": ts, "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(), "Content-Type": "application/json"}


def _num(x, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# ── signed client ─────────────────────────────────────────────────────────────
class KalshiClient:
    def __init__(self, creds: Credentials, env: str = "production", timeout: float = 10.0, session: Optional[requests.Session] = None):
        self.env = env
        self.base = base_url(env)
        self.signer = Signer(creds)
        self.timeout = timeout
        self.s = session or requests.Session()
        self.skew_s: Optional[float] = None            # server clock - local clock (seconds), from the Date header

    def _req(self, method: str, path: str, params: Optional[dict] = None, body: Optional[dict] = None) -> dict:
        """One signed request.  READS are re-sent (up to 3 tries, with Retry-After / backoff) on 429, 502-504 and network errors; ORDERS and other writes are sent exactly once - a 429 on a write is reported
        as such (the order was not placed) and what to do next is the caller's decision.  Every request first takes a slot from the process-wide limiter."""
        url = self.base + path
        data = json.dumps(body, separators=(",", ":")) if body is not None else None
        tries = 3 if method.upper() == "GET" else 1
        for attempt in range(tries):
            if not _loopback(url):
                PRIVATE_LIMITER.acquire(1.0 if method.upper() == "GET" else 0.25)
            headers = self.signer.headers(method, urlparse(url).path)                      # a fresh timestamp and signature for every try
            t0 = time.time()
            try:
                r = self.s.request(method, url, params=params, headers=headers, data=data, timeout=self.timeout)
            except requests.RequestException as e:
                if attempt + 1 < tries:
                    time.sleep(0.4 * (attempt + 1))
                    continue
                raise ApiError(0, f"network error ({type(e).__name__})") from None
            d = r.headers.get("Date")
            if d:
                try:
                    self.skew_s = parsedate_to_datetime(d).timestamp() - (t0 + time.time()) / 2
                except Exception:                          # noqa: BLE001
                    pass
            if r.status_code in RETRY_STATUS and attempt + 1 < tries:
                time.sleep(_retry_delay(r, attempt))
                continue
            break
        if r.status_code >= 400:
            msg = r.text[:300].replace("\n", " ")
            if r.status_code == 401:
                msg = "unauthorized - check the Key ID, the key file, the environment (demo and production keys differ) and that this PC's clock is correct"
            elif r.status_code == 429:
                msg += " (Kalshi's rate limit: the app already spaces out and re-sends its reads; if this keeps happening, another program, a second copy of this app or a data download is using the same key / IP)"
            raise ApiError(r.status_code, msg)
        try:
            return r.json() if r.content else {}
        except ValueError:
            raise ApiError(r.status_code, "response was not JSON") from None

    # account
    def balance(self, exchange_index: Optional[int] = None) -> dict:
        """Cash and portfolio value.  Unscoped, Kalshi adds up ALL exchange shards - but an order is funded only by the shard its market trades on, so pass that shard's ``exchange_index`` to get the cash an order can
        really use (``buying_power``).  ``breakdown`` = {shard: cash} when Kalshi provides it."""
        d = self._req("GET", "/portfolio/balance", params={"exchange_index": int(exchange_index)} if exchange_index is not None else None)
        bal = _num(d.get("balance_dollars"))
        if bal is None:
            bal = (_num(d.get("balance"), 0.0) or 0.0) / 100.0
        pv = _num(d.get("portfolio_value_dollars"))
        if pv is None:
            pv = (_num(d.get("portfolio_value"), 0.0) or 0.0) / 100.0
        breakdown = {}
        for it in d.get("balance_breakdown") or []:
            try:
                breakdown[int(it["exchange_index"])] = float(it["balance"])
            except (KeyError, TypeError, ValueError):
                pass
        if exchange_index is not None and int(exchange_index) in breakdown:
            bal = breakdown[int(exchange_index)]                   # an answer that ignored the scope still gets the right shard from the breakdown
        return {"balance": bal, "portfolio_value": pv, "buying_power": bal, "breakdown": breakdown, "exchange_index": exchange_index}

    def positions(self) -> list:
        """Unsettled market positions with a non-zero size."""
        d = self._req("GET", "/portfolio/positions", params={"limit": 1000, "settlement_status": "unsettled"})
        return [p for p in d.get("market_positions", []) if abs(_num(p.get("position_fp", p.get("position")), 0.0) or 0.0) > 1e-9]

    def open_orders(self) -> list:
        return self._req("GET", "/portfolio/orders", params={"status": "resting", "limit": 200}).get("orders", [])

    def fills(self, order_id: Optional[str] = None, ticker: Optional[str] = None, min_ts: Optional[int] = None, limit: int = 200, max_pages: int = 3) -> list:
        out, cursor = [], None
        for _ in range(max_pages):
            params = {"limit": limit}
            if order_id:
                params["order_id"] = order_id
            if ticker:
                params["ticker"] = ticker
            if min_ts:
                params["min_ts"] = int(min_ts)
            if cursor:
                params["cursor"] = cursor
            d = self._req("GET", "/portfolio/fills", params=params)
            out += d.get("fills", [])
            cursor = d.get("cursor")
            if not cursor or not d.get("fills"):
                break
        return out

    # trading
    def create_order(self, ticker: str, book_side: str, count: int, price: float, tif: str = "immediate_or_cancel", client_order_id: Optional[str] = None, exchange_index: Optional[int] = None) -> dict:
        body = {"ticker": ticker, "client_order_id": client_order_id or str(uuid.uuid4()), "side": book_side, "count": f"{int(count)}", "price": f"{price:.4f}", "time_in_force": tif,
                "self_trade_prevention_type": "taker_at_cross", "reduce_only": False}
        if exchange_index is not None:
            body["exchange_index"] = int(exchange_index)           # route straight to the shard the market lives on (omitted = Kalshi auto-routes by ticker, slower)
        return self._req("POST", "/portfolio/events/orders", body=body)

    def transfer(self, dollars: float, from_shard: int, to_shard: int, client_transfer_id: Optional[str] = None) -> dict:
        """Move cash between exchange shards of YOUR OWN account (POST /portfolio/intra_exchange_instance_transfer; amounts are in centicents = 1/10,000 dollar).  Kalshi accepts it at once and completes it
        asynchronously.  ``client_transfer_id`` makes a retry of the same request harmless (Kalshi will not execute one id twice)."""
        body = {"source": "event_contract", "destination": "event_contract", "amount": int(round(float(dollars) * 10_000)), "source_exchange_shard": int(from_shard),
                "destination_exchange_shard": int(to_shard), "client_transfer_id": client_transfer_id or str(uuid.uuid4())}
        return self._req("POST", "/portfolio/intra_exchange_instance_transfer", body=body)

    def cancel_resting(self) -> int:
        """Best effort: cancel every resting order.  The app only sends IOC orders, so there should be none; returns how many it cancelled."""
        n = 0
        for o in self.open_orders():
            oid = o.get("order_id") or o.get("id")
            if not oid:
                continue
            try:
                self._req("DELETE", f"/portfolio/events/orders/{oid}")
                n += 1
            except ApiError:
                pass
        return n


# ── broker ────────────────────────────────────────────────────────────────────
_LOOKUP = object()


class KalshiBroker:
    """Real orders.  Every buy is an IOC limit order, so nothing ever rests on the book; the exact fills are read back from GET /portfolio/fills."""
    mode = "live"
    venue = "kalshi"

    def __init__(self, client: KalshiClient, fill_wait_s: float = 6.0, exchange_index=_LOOKUP):
        self.c = client
        self.fill_wait_s = fill_wait_s
        self._idx = exchange_index                     # default: look up the shard of the BTC 15-minute series on first use; None = unknown (Kalshi then auto-routes by ticker and the total balance is used)

    @property
    def exchange_index(self) -> Optional[int]:
        """The exchange shard BTC 15-minute orders are routed to and funded from (public series data, looked up once)."""
        if self._idx is _LOOKUP:
            try:
                self._idx = KalshiQuotes(base=self.c.base).exchange_index()
            except Exception:                          # noqa: BLE001
                self._idx = None
        return self._idx

    def buying_power(self) -> Optional[float]:
        return self.c.balance(self.exchange_index)["buying_power"]

    def fund(self, shortfall: float, wait_s: float = 15.0) -> dict:
        """What the Kalshi app does before an order: move ``shortfall`` dollars of cash from your other exchange shards onto the shard BTC 15-minute orders are funded from (largest balance first).  Cash only ever
        moves between shards of your own account.  Waits up to ``wait_s`` for it to show up.  Returns {moved, arrived, balance, moves=[(shard, dollars)]}; ``moved`` is 0 when no other shard holds cash."""
        dest = self.exchange_index
        if dest is None:
            raise BrokerError("auto-fund: the exchange shard of the BTC 15-minute series is unknown, so there is nothing to fund")
        tot = self.c.balance()
        have = self.c.balance(dest)["buying_power"]
        left, moves = round(float(shortfall), 2), []
        for cash, shard in sorted(((v, i) for i, v in tot["breakdown"].items() if i != dest and v >= 0.01), reverse=True):
            take = round(min(left, cash), 2)
            if take < 0.01:
                continue
            tid = str(uuid.uuid4())
            try:
                try:
                    self.c.transfer(take, shard, dest, tid)
                except ApiError as e:
                    if e.status != 0 and e.status < 500:
                        raise
                    self.c.transfer(take, shard, dest, tid)           # network trouble or a 5xx: the same id again - Kalshi will not execute it twice
            except ApiError as e:
                raise BrokerError(f"cash transfer from shard {shard} to shard {dest} failed: {e.message}") from None
            moves.append((shard, take))
            left = round(left - take, 2)
            if left < 0.01:
                break
        moved = round(sum(t for _, t in moves), 2)
        if not moves:
            return dict(moved=0.0, arrived=False, balance=have, moves=[])
        deadline, bal = time.time() + wait_s, have
        while True:
            bal = self.c.balance(dest)["buying_power"]
            if bal >= have + moved - 0.005 or time.time() >= deadline:
                return dict(moved=moved, arrived=bal >= have + moved - 0.005, balance=bal, moves=moves)
            time.sleep(0.5)

    def funds_message(self) -> str:
        """Why an order was refused for lack of cash, with the numbers (best effort: a failing balance read never hides the original error)."""
        idx = self.exchange_index
        if idx is None:                                # this broker does not route by shard, but the message should still name the shard Kalshi checks
            try:
                idx = KalshiQuotes(base=self.c.base).exchange_index()
            except Exception:                          # noqa: BLE001
                idx = None
        try:
            tot = self.c.balance()
            mine = self.c.balance(idx) if idx is not None else tot
            where = ", ".join(f"shard {i}: ${v:,.2f}" for i, v in sorted(tot["breakdown"].items()))
            if idx is None:
                return f"Kalshi rejected the order for insufficient balance (cash ${tot['buying_power']:,.2f}) - add funds or lower the stake."
            return (f"Kalshi rejected the order for insufficient balance. BTC 15-minute markets trade on exchange shard {idx}, which holds ${mine['buying_power']:,.2f} of cash; your total cash across all shards is "
                    f"${tot['buying_power']:,.2f}" + (f" ({where})" if where else "") + f". An API order is funded only by the shard it trades on (the Kalshi app moves cash between shards automatically, the API does not): "
                    f"transfer cash to shard {idx} on the Exchange Indexes page of your Kalshi account settings, then the next signal will trade.")
        except ApiError:
            return "Kalshi rejected the order for insufficient balance (cash on the exchange shard this market trades on is too small; the balance could not be read to say more)."

    @staticmethod
    def v2_order(side: str, max_cost: float) -> tuple[str, float]:
        """('bid' | 'ask', YES-book price) for buying ``side`` without paying more than ``max_cost`` per contract of that side.
        YES: bid at floor_cent(max_cost).  NO: buying NO at q = selling YES at 1 - q, so ask at ceil_cent(1 - max_cost) (rounded so the NO cost never exceeds max_cost)."""
        if side.upper() == "YES":
            return "bid", min(cent_floor(max_cost), 0.99)
        return "ask", max(cent_ceil(1.0 - max_cost), 0.01)

    def buy(self, ticker: str, side: str, contracts: int, max_cost: float, quoted_cost: float) -> Optional[Fill]:
        side = side.upper()
        want = "yes" if side == "YES" else "no"
        book_side, price = self.v2_order(side, max_cost)
        if not (0.01 <= price <= 0.99):
            raise NotPlaced(f"limit price {price:.2f} is outside 1c-99c")
        t0 = time.time()
        try:
            resp = self.c.create_order(ticker, book_side, int(contracts), price, exchange_index=self.exchange_index)
        except ApiError as e:
            if e.status == 429:
                raise NotPlaced("rate limited by Kalshi (HTTP 429) - the order was not sent") from None
            if 400 <= e.status < 500 and e.status not in (408, 409):
                if "insufficient_balance" in e.message.lower() or "insufficient balance" in e.message.lower():
                    raise InsufficientFunds(self.funds_message()) from None
                raise BrokerError(f"order rejected by Kalshi: {e.message}") from None
            return self._recover(ticker, want, t0, max_cost, f"order submission failed ({e.status})")
        oid = str(resp.get("order_id", ""))
        if not oid:
            return self._recover(ticker, want, t0, max_cost, "order accepted but no order id returned")
        filled = _num(resp.get("fill_count"), 0.0) or 0.0
        if filled <= 1e-9:
            return None                                                      # IOC found nothing at the limit: flat
        note = f"order {oid} {book_side}@{price:.2f} fill_count={filled:g} avg_resp={resp.get('average_fill_price')} fee_resp={resp.get('average_fee_paid')} ts_ms={resp.get('ts_ms')}"
        try:
            fills = self._wait_fills(oid, filled)
        except ApiError:
            fills = []
        if fills:
            return self._fill_from_fills(fills, want, max_cost, oid, note)
        return self._fill_from_response(resp, side, want, filled, max_cost, oid, note)

    # ---- helpers
    def _wait_fills(self, order_id: str, expected: float) -> list:
        deadline = time.time() + self.fill_wait_s
        fills: list = []
        while True:
            fills = self.c.fills(order_id=order_id)
            if sum(_num(f.get("count_fp", f.get("count")), 0.0) or 0.0 for f in fills) >= expected - 1e-6 or time.time() >= deadline:
                return fills
            time.sleep(0.3)

    def _fill_from_fills(self, fills: list, want: str, max_cost: float, oid: str, note: str) -> Fill:
        qty = cost = fee = 0.0
        for f in fills:
            side = f.get("outcome_side") or f.get("side")
            if side not in (None, want):
                raise BrokerError(f"fill on the wrong side (expected {want}, exchange says {side}) for order {oid} - check the account")
            n = _num(f.get("count_fp", f.get("count")), 0.0) or 0.0
            px = _num(f.get("no_price_dollars" if want == "no" else "yes_price_dollars"))
            if px is None:
                raise BrokerError(f"fill without a price for order {oid} - check the account")
            qty += n
            cost += n * px
            fee += _num(f.get("fee_cost"), 0.0) or 0.0
        if qty <= 1e-9:
            raise BrokerError(f"order {oid} reported fills but none carry a size - check the account")
        price = cost / qty
        if price > max_cost + 1e-6:
            raise BrokerError(f"fill price {price:.4f} is above the limit {max_cost:.2f} for order {oid} - check the account")
        return Fill(qty=int(round(qty)), price=float(price), fee=float(fee), order_id=oid, state="FILLED", note=note)

    def _fill_from_response(self, resp: dict, side: str, want: str, filled: float, max_cost: float, oid: str, note: str) -> Fill:
        """Fallback when the fills endpoint is unavailable: the order response's own average price / fee.  The price is in YES-book terms; a NO buy tries both readings and keeps the one inside the limit."""
        avg = _num(resp.get("average_fill_price"))
        if avg is None:
            raise BrokerError(f"order {oid} filled {filled:g} but the exchange gave no price - check the account")
        cands = [avg] if side == "YES" else [1.0 - avg, avg]
        price = next((c for c in cands if 0.005 <= c <= max_cost + 1e-6), None)
        if price is None:
            raise BrokerError(f"fill price {avg} violates the limit {max_cost:.2f} under both price conventions (order {oid})")
        fee = (_num(resp.get("average_fee_paid"), 0.0) or 0.0) * filled
        return Fill(qty=int(round(filled)), price=float(price), fee=float(fee), order_id=oid, state="FILLED", note=note + " [from the order response: fills endpoint unavailable]")

    def _recover(self, ticker: str, want: str, t0: float, max_cost: float, why: str) -> Optional[Fill]:
        """Submission outcome unknown (network error / 5xx): look for executions on this market since the attempt; none seen = still unknown = stop trading."""
        deadline = time.time() + self.fill_wait_s
        while True:
            try:
                fills = [f for f in self.c.fills(ticker=ticker, min_ts=int(t0) - 2) if (f.get("outcome_side") or f.get("side")) in (None, want)]
            except ApiError:
                fills = []
            if fills:
                oid = str(fills[0].get("order_id", ""))
                return self._fill_from_fills([f for f in fills if str(f.get("order_id", "")) == oid] or fills, want, max_cost, oid, f"recovered after: {why}")
            if time.time() >= deadline:
                raise BrokerError(f"{why}; no fills seen on {ticker} afterwards - order state unknown, check the Kalshi account")
            time.sleep(0.5)

    def cancel_all(self) -> None:
        self.c.cancel_resting()

    def order_path_test(self, ticker: str) -> dict:
        """Safe end-to-end check of signing and the V2 order schema: a 1-contract IOC BID at 1c.  It cannot rest and fills only if someone sells YES at 1c (cost: at most one cent)."""
        try:
            r = self.c.create_order(ticker, "bid", 1, 0.01, exchange_index=self.exchange_index)
        except ApiError as e:
            if 400 <= e.status < 500 and ("insufficient_balance" in e.message.lower() or "insufficient balance" in e.message.lower()):
                raise InsufficientFunds(self.funds_message()) from None
            raise
        return {"order_id": r.get("order_id"), "fill_count": _num(r.get("fill_count"), 0.0), "remaining_count": _num(r.get("remaining_count"), 0.0)}
