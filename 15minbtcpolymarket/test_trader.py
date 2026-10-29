"""Offline tests for btc15m/trader (no network except a loopback mock server, no real credentials).
Run:  python test_trader.py
"""
import base64
import json
import os
import tempfile
import threading
from pathlib import Path
import time
from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

from trader import kalshi
from trader.backtest import BacktestConfig, PROB_MODES, run_backtest
from trader.broker import BrokerError, Fill, NotPlaced
from trader.data import extract_signals
from trader.fees import fee_total, kalshi_fee_total
from trader.kalshi import ApiError, Book, CredentialError, Credentials, KalshiBroker, KalshiClient, Signer, load_credentials
from trader.live import LiveConfig, LiveEngine, PaperBroker
from trader.probmodel import SignalStats
from trader.metrics import compute_metrics
from trader.strategy import StrategyParams, evaluate_quote, evaluate_quote_all, fmt_int_list, kelly_full, parse_int_list, size_bet

# throwaway keys for tests only
_RSA = rsa.generate_private_key(public_exponent=65537, key_size=2048)
TEST_PEM = _RSA.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()
TEST_KEY_ID = "12345678-1234-1234-1234-123456789abc"
CREDS = Credentials(TEST_KEY_ID, TEST_PEM)
PSS = padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH)


# ── fees / strategy / kelly ───────────────────────────────────────────────────
def test_fees():
    assert kalshi_fee_total(100, 0.5) == 1.75
    assert kalshi_fee_total(1, 0.5) == 0.02                      # ceil to the next cent
    assert fee_total("Kalshi", 0, 0.5) == 0 and fee_total("Kalshi", 10, 1.0) == 0
    try:
        fee_total("Polymarket US", 10, 0.5)                      # the retired venue's schedule must not be used silently
        assert False
    except ValueError:
        pass


def test_parse_int_list():
    assert parse_int_list("22,23") == (22, 23)
    assert parse_int_list("20-21, 23") == (20, 21, 23)
    for bad in ("", "24", "a"):
        try:
            parse_int_list(bad)
            assert False, bad
        except ValueError:
            pass


def test_evaluate_quote():
    p = StrategyParams()
    s = evaluate_quote(0.15, 0.16, 12, 22, p)                    # YES is the underdog -> buy YES at the ask
    assert s and s.side == "YES" and abs(s.cost - 0.16) < 1e-9
    s = evaluate_quote(0.84, 0.85, 13, 23, p)                    # NO is the underdog -> cost = 1 - yes_bid
    assert s and s.side == "NO" and abs(s.cost - 0.16) < 1e-9
    assert evaluate_quote(0.15, 0.16, 12, 21, p) is None         # wrong hour
    assert evaluate_quote(0.15, 0.16, 11, 22, p) is None         # wrong minute
    assert evaluate_quote(0.40, 0.41, 12, 22, p) is None         # underdog cost 0.41 > 0.30
    assert evaluate_quote(0.01, 0.02, 12, 22, p) is None         # underdog cost 0.02 < 0.05
    assert evaluate_quote(0.0, 1.0, 12, 22, p) is None           # empty book
    assert evaluate_quote(0.10, 0.30, 12, 22, p) is None         # spread too wide


def test_kelly_and_sizing():
    assert abs(kelly_full(0.6, 0.5) - 0.2) < 1e-12
    assert kelly_full(0.4, 0.5) == 0.0
    z = size_bet(10_000, 0.20, 0.25, "Kalshi", 0.25, 0.02, 1000, 0.005)     # p below price -> no bet
    assert z.contracts == 0 and "edge" in z.reason
    s = size_bet(10_000, 0.40, 0.20, "Kalshi", 0.25, 0.02, 100_000, 0.005)
    assert s.f_full > 0.2 and abs(s.f_used - 0.02) < 1e-12                          # quarter-Kelly would be >5% -> capped at 2%
    assert s.contracts > 0 and s.stake <= 0.02 * 10_000 + 1e-6
    assert s.contracts == int(s.contracts)
    s2 = size_bet(100, 0.40, 0.20, "Kalshi", 0.25, 0.02, 1000, 0.005)       # $2 budget at 20c -> 8 contracts
    assert 6 <= s2.contracts <= 9 and s2.stake <= 2.0 + 1e-9


# ── Kalshi: order book, V2 order prices, credentials, signing ─────────────────
def test_book_depth_and_walk():
    d = {"orderbook_fp": {"yes_dollars": [["0.30", "10"], ["0.71", "5"], ["0.70", "8"]], "no_dollars": [["0.20", "3"], ["0.27", "9"], ["0.25", "4"]]}}
    b = Book.from_response(d)
    assert b.top() == (0.71, 0.73) and b.yes_bids[0] == (0.71, 5.0)                          # YES ask = 1 - best NO bid
    assert [round(p, 2) for p, _ in b.asks("YES")] == [0.73, 0.75, 0.80]                       # buying YES: 1 - NO bids, cheapest first
    assert [round(p, 2) for p, _ in b.asks("NO")] == [0.29, 0.30, 0.70]                        # buying NO: 1 - YES bids
    assert b.walk("YES", 5, 0.75) == (5.0, 0.73) and b.walk("YES", 20, 0.75) == (13.0, (9 * 0.73 + 4 * 0.75) / 13)      # only the levels inside the limit count
    assert b.walk("YES", 5, 0.72) == (0.0, None)
    assert Book.from_response({"orderbook": {"yes": [[44, 2]], "no": [[50, 1]]}}).top() == (0.44, 0.50)                     # older cent format
    e = Book.from_response({"orderbook_fp": {"yes_dollars": [], "no_dollars": []}})
    assert e.top() == (0.0, 1.0) and Book.from_response({}) is None and Book.from_response({"orderbook_fp": {"yes_dollars": [["x", "1"]]}}) is None


def test_v2_order_prices():
    assert KalshiBroker.v2_order("YES", 0.237) == ("bid", 0.23)                              # buy YES: bid at the highest cent not above the cap
    side, px = KalshiBroker.v2_order("NO", 0.237)                                            # buy NO at <= 0.237  ==  sell YES at >= 0.763 -> 0.77
    assert side == "ask" and px == 0.77
    for mc in (0.05, 0.1, 0.19, 0.2999, 0.5):                                                # never pay more than the cap, either side
        assert KalshiBroker.v2_order("YES", mc)[1] <= mc + 1e-9
        assert 1 - KalshiBroker.v2_order("NO", mc)[1] <= mc + 1e-9


def test_credentials_loading_and_secrecy():
    d = Path(tempfile.mkdtemp())
    pem_only = d / "k.pem"
    pem_only.write_text(TEST_PEM)
    c = load_credentials(pem_only, TEST_KEY_ID)
    assert c.key_id == TEST_KEY_ID and TEST_PEM not in repr(c) and TEST_PEM not in str(c) and "hidden" in repr(c)
    both = d / "both.txt"
    both.write_text(f"API Key ID: {TEST_KEY_ID}\r\n\r\n{TEST_PEM}\r\n")
    assert load_credentials(both).key_id == TEST_KEY_ID                                       # the id may sit in the same file
    ed = ed25519.Ed25519PrivateKey.generate().private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()
    (d / "ed.pem").write_text(ed)
    assert load_credentials(d / "ed.pem", TEST_KEY_ID).key_id == TEST_KEY_ID                  # Ed25519 keys work too
    junk = d / "junk.pem"
    junk.write_text("-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----\n")
    for path, kid in ((d / "missing.pem", TEST_KEY_ID), (d / "k.pem", ""), (d / "k.pem", "not-a-uuid"), (junk, TEST_KEY_ID)):
        try:
            load_credentials(path, kid)
            assert False, (path, kid)
        except CredentialError as e:
            assert TEST_PEM not in str(e) and "AAAA" not in str(e)
    (d / "nokey.txt").write_text("hello")
    try:
        load_credentials(d / "nokey.txt", TEST_KEY_ID)
        assert False
    except CredentialError:
        pass


def test_signature_verifies():
    sg = Signer(CREDS)
    h = sg.headers("get", "/trade-api/v2/portfolio/balance", ts_ms=1_700_000_000_123)
    assert h["KALSHI-ACCESS-KEY"] == TEST_KEY_ID and h["KALSHI-ACCESS-TIMESTAMP"] == "1700000000123"
    _RSA.public_key().verify(base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]), b"1700000000123GET/trade-api/v2/portfolio/balance", PSS, hashes.SHA256())      # RSA-PSS / SHA-256 / salt = digest length
    edk = ed25519.Ed25519PrivateKey.generate()
    ed_pem = edk.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()
    h2 = Signer(Credentials(TEST_KEY_ID, ed_pem)).headers("POST", "/trade-api/v2/portfolio/events/orders", ts_ms=5)
    edk.public_key().verify(base64.b64decode(h2["KALSHI-ACCESS-SIGNATURE"]), b"5POST/trade-api/v2/portfolio/events/orders")


class MockKalshi(BaseHTTPRequestHandler):
    """A signature-verifying stand-in for Kalshi: public market data, balance / positions / orders / fills, and V2 IOC orders with selectable behaviour (class attribute ``mode``)."""
    created: list = []                 # order bodies received
    fills: list = []                   # fills the 'exchange' holds
    paths: list = []                   # every signed path (without query) seen
    mode = "fill"                      # fill | partial | nofill | reject400 | rate429 | error500 | error500_filled | wrong_side | fills_down
    yes_bid, yes_ask = 0.14, 0.15
    ticker = "KXBTC15M-MOCK-00"
    positions: list = []               # market_positions the 'account' holds
    resting: list = []                 # resting orders the 'account' holds
    shards: dict = {0: 950.5}          # cash per exchange shard; an order is funded ONLY by the shard its market trades on (like the real exchange)
    transfers: list = []               # accepted intra-account transfer bodies
    transfer_mode = "ok"               # ok | reject400
    transfer_lag = 0                   # balance reads before a transfer shows up on the destination shard (Kalshi completes transfers asynchronously)
    throttle_gets = 0                  # the next N signed GETs are answered with HTTP 429 (+ Retry-After: 0)
    _pending: list = []
    _transfer_ids: set = set()
    series_index = 0                   # the shard the BTC 15-minute series trades on (the real one: 2)
    pub = _RSA.public_key()

    def log_message(self, *a):
        pass

    def _auth_ok(self, method):
        try:
            path = self.path.split("?")[0]
            assert self.headers["KALSHI-ACCESS-KEY"] == TEST_KEY_ID and path.startswith("/trade-api/v2/")
            self.pub.verify(base64.b64decode(self.headers["KALSHI-ACCESS-SIGNATURE"]), f"{self.headers['KALSHI-ACCESS-TIMESTAMP']}{method}{path}".encode(), PSS, hashes.SHA256())      # the path only, never the query
            self.paths.append(path)
            return True
        except Exception:
            return False

    def _send(self, code, obj, headers=None):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse
        u = urlparse(self.path)
        route, q = u.path[len("/trade-api/v2"):], parse_qs(u.query)
        if route == "/exchange/status":
            return self._send(200, {"exchange_active": True, "trading_active": True, "exchange_index_statuses": [{"exchange_index": 0, "description": "Default"}, {"exchange_index": 2, "description": "Crypto & Commodities"}]})
        if route == "/series/KXBTC15M":
            return self._send(200, {"series": {"ticker": "KXBTC15M", **({} if self.series_index is None else {"exchange_index": self.series_index})}})
        if route == "/markets":
            w = int(time.time()) // 900 * 900
            return self._send(200, {"markets": [{"ticker": self.ticker, "open_time": datetime.fromtimestamp(w, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}]})
        if route == f"/markets/{self.ticker}":
            return self._send(200, {"market": {"ticker": self.ticker, "status": "active", "yes_bid_dollars": f"{self.yes_bid:.4f}", "yes_ask_dollars": f"{self.yes_ask:.4f}",
                                               "price_ranges": [{"start": "0.0000", "end": "0.1000", "step": "0.0010"}, {"start": "0.1000", "end": "0.9000", "step": "0.0100"}, {"start": "0.9000", "end": "1.0000", "step": "0.0010"}]}})
        if route == f"/markets/{self.ticker}/orderbook":
            return self._send(200, {"orderbook_fp": {"yes_dollars": [[f"{self.yes_bid:.4f}", "500.00"]], "no_dollars": [[f"{1 - self.yes_ask:.4f}", "700.00"]]}})
        if not self._auth_ok("GET"):
            return self._send(401, {"error": "bad signature"})
        if MockKalshi.throttle_gets > 0:
            MockKalshi.throttle_gets -= 1
            return self._send(429, {"error": {"code": "too_many_requests", "message": "too many requests"}}, {"Retry-After": "0"})
        if route == "/portfolio/balance":
            for p in list(self._pending):                                                      # asynchronous transfers arriving
                p[2] -= 1
                if p[2] <= 0:
                    self.shards[p[0]] = self.shards.get(p[0], 0.0) + p[1]
                    self._pending.remove(p)
            tot = sum(self.shards.values())
            bal = self.shards.get(int(q["exchange_index"][0]), 0.0) if q.get("exchange_index") else tot
            return self._send(200, {"balance": int(round(bal * 100)), "balance_dollars": f"{bal:.4f}", "portfolio_value": 120000, "updated_ts": 1,
                                    "balance_breakdown": [{"exchange_index": i, "balance": f"{v:.4f}"} for i, v in sorted(self.shards.items())]})
        if route == "/portfolio/orders":
            return self._send(200, {"orders": self.resting, "cursor": ""})
        if route == "/portfolio/positions":
            return self._send(200, {"market_positions": self.positions, "event_positions": [], "cursor": ""})
        if route == "/portfolio/fills":
            if self.mode == "fills_down":
                return self._send(500, {"error": "down"})
            out = [f for f in self.fills if (not q.get("order_id") or f["order_id"] == q["order_id"][0]) and (not q.get("ticker") or f["ticker"] in q["ticker"][0].split(","))
                   and (not q.get("min_ts") or f.get("ts", 0) >= int(q["min_ts"][0]))]
            return self._send(200, {"fills": out, "cursor": ""})
        self._send(404, {})

    def do_DELETE(self):
        if not self._auth_ok("DELETE"):
            return self._send(401, {"error": "bad signature"})
        self._send(200, {"order_id": self.path.rsplit("/", 1)[1], "reduced_by": "0.00"})

    def do_POST(self):
        if not self._auth_ok("POST"):
            return self._send(401, {"error": "bad signature"})
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        route = self.path.split("?")[0][len("/trade-api/v2"):]
        if route == "/portfolio/intra_exchange_instance_transfer":
            if self.transfer_mode == "reject400":
                return self._send(400, {"error": {"code": "invalid_transfer", "message": "transfer refused"}})
            assert body["source"] == "event_contract" and body["destination"] == "event_contract" and isinstance(body["amount"], int) and body["client_transfer_id"]
            if body["client_transfer_id"] in self._transfer_ids:                              # the same id is never executed twice
                return self._send(200, {"transfer_id": "tr-dup"})
            self._transfer_ids.add(body["client_transfer_id"])
            amt, src, dst = body["amount"] / 10_000.0, int(body["source_exchange_shard"]), int(body["destination_exchange_shard"])
            if amt > self.shards.get(src, 0.0) + 1e-9:
                return self._send(400, {"error": {"code": "insufficient_balance", "message": "insufficient balance"}})
            self.shards[src] -= amt
            self._pending.append([dst, amt, self.transfer_lag])
            self.transfers.append(body)
            for p in list(self._pending):                                                      # no lag: it is there at once
                if p[2] <= 0:
                    self.shards[p[0]] = self.shards.get(p[0], 0.0) + p[1]
                    self._pending.remove(p)
            return self._send(200, {"transfer_id": f"tr-{len(self.transfers)}"})
        if route != "/portfolio/events/orders":
            return self._send(404, {})
        self.created.append(body)
        assert body["time_in_force"] == "immediate_or_cancel" and body["self_trade_prevention_type"] == "taker_at_cross" and body["side"] in ("bid", "ask")
        n, px = int(body["count"]), float(body["price"])
        cost = n * px if body["side"] == "bid" else n * (1.0 - px)
        if cost > self.shards.get(int(body.get("exchange_index", self.series_index or 0)), 0.0) + 1e-9:         # collateral is checked inside the shard the order routes to
            return self._send(400, {"error": {"code": "insufficient_balance", "message": "insufficient balance"}})
        oid = f"ord-{len(self.created)}"
        if self.mode == "rate429":
            return self._send(429, {"error": "slow down"})
        if self.mode == "reject400":
            return self._send(400, {"error": {"code": "invalid_order", "message": "market closed"}})
        fill_n = {"nofill": 0, "partial": n // 2}.get(self.mode, n)
        buys_yes = body["side"] == "bid"
        yes_px = px - 0.02 if buys_yes else px + 0.02                       # the exchange fills 2c better than our limit (YES-book price)
        if fill_n and self.mode != "error500":
            side_out = ("no" if buys_yes else "yes") if self.mode == "wrong_side" else ("yes" if buys_yes else "no")
            self.fills.append({"fill_id": f"f{len(self.fills)}", "order_id": oid, "ticker": body["ticker"], "outcome_side": side_out, "book_side": body["side"], "count_fp": f"{fill_n:.2f}",
                               "yes_price_dollars": f"{yes_px:.4f}", "no_price_dollars": f"{1 - yes_px:.4f}", "is_taker": True, "fee_cost": "0.1200", "created_time": datetime.now(timezone.utc).isoformat(), "ts": int(time.time())})
        if self.mode in ("error500", "error500_filled"):
            return self._send(500, {"error": "internal"})
        resp = {"order_id": oid, "client_order_id": body["client_order_id"], "fill_count": f"{fill_n:.2f}", "remaining_count": f"{n - fill_n:.2f}", "ts_ms": int(time.time() * 1000)}
        if fill_n:
            resp.update(average_fill_price=f"{yes_px:.4f}", average_fee_paid="0.0030")
        return self._send(201, resp)


def _mock_server():
    MockKalshi.created, MockKalshi.fills, MockKalshi.paths, MockKalshi.mode = [], [], [], "fill"
    MockKalshi.positions, MockKalshi.resting = [], []
    MockKalshi.shards, MockKalshi.series_index = {0: 950.5}, 0
    MockKalshi.transfers, MockKalshi.transfer_mode, MockKalshi.transfer_lag, MockKalshi._pending, MockKalshi._transfer_ids = [], "ok", 0, [], set()
    MockKalshi.throttle_gets = 0
    MockKalshi.yes_bid, MockKalshi.yes_ask = 0.14, 0.15
    srv = ThreadingHTTPServer(("127.0.0.1", 0), MockKalshi)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/trade-api/v2"


def _client(url, creds=CREDS):
    old = os.environ.get("KALSHI_API_BASE")
    os.environ["KALSHI_API_BASE"] = url
    try:
        return KalshiClient(creds, "demo", timeout=3)
    finally:
        if old is None:
            os.environ.pop("KALSHI_API_BASE", None)
        else:
            os.environ["KALSHI_API_BASE"] = old


def test_client_and_broker_against_mock():
    srv, url = _mock_server()
    try:
        c = _client(url)
        MockKalshi.positions = [{"ticker": "KXBTC15M-OLD-00", "position_fp": "-12.00"}, {"ticker": "KXETH-X", "position_fp": "0.00"}]
        b = c.balance()
        assert b["buying_power"] == 950.5 and b["portfolio_value"] == 1200.0                     # signature accepted by the server
        assert [p["ticker"] for p in c.positions()] == ["KXBTC15M-OLD-00"] and c.open_orders() == []         # zero-size positions are not positions
        assert "/trade-api/v2/portfolio/balance" in MockKalshi.paths and all("?" not in p for p in MockKalshi.paths)
        bad = _client(url, Credentials(TEST_KEY_ID, rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()))
        try:
            bad.balance()
            assert False
        except ApiError as e:
            assert e.status == 401 and TEST_PEM not in str(e)
        br = KalshiBroker(c, fill_wait_s=2)
        f = br.buy(MockKalshi.ticker, "YES", 40, 0.20, 0.17)
        sent = MockKalshi.created[-1]
        assert sent["side"] == "bid" and sent["price"] == "0.2000" and sent["count"] == "40" and sent["ticker"] == MockKalshi.ticker and sent["client_order_id"]
        assert f.qty == 40 and abs(f.price - 0.18) < 1e-9 and abs(f.fee - 0.12) < 1e-9 and f.order_id == "ord-1"      # limit 0.20, mock fills 2c better -> 0.18; the fee comes from the fills
        assert "ord-1" in f.note and br.buying_power() == 950.5
        t = br.order_path_test(MockKalshi.ticker)                                                # the 1-contract IOC bid at 1c used as an end-to-end schema check
        assert MockKalshi.created[-1]["price"] == "0.0100" and MockKalshi.created[-1]["count"] == "1" and t["order_id"]
    finally:
        srv.shutdown()


def test_no_side_price_convention_end_to_end():
    srv, url = _mock_server()
    try:
        f = KalshiBroker(_client(url), fill_wait_s=2).buy(MockKalshi.ticker, "NO", 10, 0.20, 0.17)
        sent = MockKalshi.created[-1]
        assert sent["side"] == "ask" and sent["price"] == "0.8000"                                # buying NO at <= 0.20 == selling YES at >= 0.80
        assert abs(f.price - 0.18) < 1e-9 and f.qty == 10                                         # mock filled at YES 0.82 == NO 0.18 <= our 0.20 cap
    finally:
        srv.shutdown()


def test_broker_failure_modes():
    srv, url = _mock_server()
    try:
        br = KalshiBroker(_client(url), fill_wait_s=1.2)
        MockKalshi.mode = "nofill"
        assert br.buy(MockKalshi.ticker, "YES", 5, 0.2, 0.17) is None                            # IOC found nothing -> flat, no error
        MockKalshi.mode = "partial"
        f = br.buy(MockKalshi.ticker, "YES", 9, 0.2, 0.17)
        assert f.qty == 4                                                                        # partial fills are booked as filled, not as ordered
        MockKalshi.mode = "rate429"
        try:
            br.buy(MockKalshi.ticker, "YES", 5, 0.2, 0.17)
            assert False
        except NotPlaced:
            pass                                                                                 # definitely not placed: the engine skips the signal and carries on
        MockKalshi.mode = "reject400"
        try:
            br.buy(MockKalshi.ticker, "YES", 5, 0.2, 0.17)
            assert False
        except BrokerError as e:
            assert "market closed" in str(e)
        MockKalshi.mode = "wrong_side"                                                           # the exchange says we bought the other side: never book it
        try:
            br.buy(MockKalshi.ticker, "NO", 5, 0.2, 0.17)
            assert False
        except BrokerError as e:
            assert "wrong side" in str(e)
        MockKalshi.fills.clear()
        MockKalshi.mode = "error500"                                                             # submission outcome unknown and no fills seen: stop trading
        try:
            br.buy(MockKalshi.ticker, "YES", 5, 0.2, 0.17)
            assert False
        except BrokerError as e:
            assert "unknown" in str(e)
        MockKalshi.mode = "error500_filled"                                                      # the order DID execute although the response was an error: recovered from the fills
        f = br.buy(MockKalshi.ticker, "YES", 7, 0.2, 0.17)
        assert f.qty == 7 and "recovered" in f.note
        MockKalshi.mode = "fills_down"                                                           # fills endpoint unavailable: fall back to the order response (YES-book price)
        f = br.buy(MockKalshi.ticker, "YES", 6, 0.2, 0.17)
        assert f.qty == 6 and abs(f.price - 0.18) < 1e-9 and "fills endpoint unavailable" in f.note
        f = br.buy(MockKalshi.ticker, "NO", 6, 0.2, 0.17)                                        # NO: average_fill_price 0.82 is the YES-book price -> NO cost 0.18 (inside the 0.20 limit)
        assert f.qty == 6 and abs(f.price - 0.18) < 1e-9
    finally:
        srv.shutdown()


def test_fill_price_above_the_limit_is_refused():
    srv, url = _mock_server()
    try:
        c = _client(url)
        br = KalshiBroker(c, fill_wait_s=1)
        MockKalshi.mode = "fill"
        orig = MockKalshi.do_POST
        def hostile(self):                                                                       # an exchange answering outside our limit would mean we misread the order semantics
            body_len = int(self.headers["Content-Length"])
            raw = self.rfile.read(body_len)
            body = json.loads(raw)
            self.created.append(body)
            self.fills.append({"fill_id": "x", "order_id": "ord-bad", "ticker": body["ticker"], "outcome_side": "yes", "book_side": "bid", "count_fp": "5.00", "yes_price_dollars": "0.2500",
                               "no_price_dollars": "0.7500", "is_taker": True, "fee_cost": "0.10"})
            self._send(201, {"order_id": "ord-bad", "fill_count": "5.00", "remaining_count": "0.00", "ts_ms": 1})
        MockKalshi.do_POST = hostile
        try:
            try:
                br.buy(MockKalshi.ticker, "YES", 5, 0.20, 0.17)
                assert False
            except BrokerError as e:
                assert "above the limit" in str(e)
        finally:
            MockKalshi.do_POST = orig
    finally:
        srv.shutdown()


# ── backtest / metrics ────────────────────────────────────────────────────────
def _rows(outcomes, hour=22, start=datetime(2026, 9, 1, tzinfo=timezone.utc)):
    """One qualifying signal per day: YES underdog bid .14/ask .15 at minute 12; ``outcomes`` = list of yes results."""
    recs = []
    for i, y in enumerate(outcomes):
        open_ts = int(start.timestamp()) + i * 86400 + hour * 3600
        recs.append(dict(ticker=f"T{i}", open_ts=open_ts, minute=12, bid=0.14, ask=0.15, yes=int(y)))
    return pd.DataFrame(recs)


def test_backtest_arithmetic_and_metrics():
    outcomes = [1, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    rows = _rows(outcomes)
    cfg = BacktestConfig(start=date(2026, 9, 1), end=date(2026, 9, 12), capital=10_000, fee_model="Kalshi", slippage=0.0,
                         kelly_mult=1.0, max_frac=0.05, min_edge=0.0, prob_mode=PROB_MODES[1], fixed_p=0.40, max_contracts=10**6)
    r = run_backtest(rows, cfg)
    t = r.trades
    assert len(t) == 10
    # first trade: budget 5% of 10k = 500 -> n contracts at 0.15 + fee; pnl on a win = n*(1-.15) - fee
    n0 = t.contracts.iloc[0]
    assert n0 == int(n0) and 3000 <= n0 <= 3333
    fee0 = kalshi_fee_total(n0, 0.15)
    assert abs(t.pnl.iloc[0] - (n0 * 0.85 - fee0)) < 1e-6 and t.outcome.iloc[0] == "WIN"
    assert abs(t.equity_after.iloc[-1] - (10_000 + t.pnl.sum())) < 1e-6
    m = r.metrics
    assert m["trades"] == 10 and m["wins"] == 4 and m["losses"] == 6
    assert m["max_consec_wins"] == 3 and m["max_consec_losses"] == 4
    assert m["max_drawdown"] < 0 and abs(m["total_return"] - (m["final_equity"] / 10_000 - 1)) < 1e-12
    eq = r.equity.equity.values
    assert abs(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min() - m["max_drawdown"]) < 1e-12
    assert m["profit_factor"] > 0 and np.isfinite(m["sharpe"])


def test_backtest_expanding_p_has_no_lookahead():
    # 40 losing signals then 40 winning: the learned edge must stay <= 0 during the losses (no bets) and only turn positive after wins accumulate
    rows = _rows([0] * 40 + [1] * 40)
    cfg = BacktestConfig(start=date(2026, 9, 1), end=date(2026, 12, 31), capital=10_000, fee_model="Kalshi", slippage=0.0)
    r = run_backtest(rows, cfg)
    assert 0 < len(r.trades) < 40
    day = (r.trades.entry_ts - datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp()) / 86400
    assert day.min() > 40 and r.trades.p_used.iloc[0] < 0.30                 # first bet only after the wins, and with a modest (not hindsight 50%) p
    full = run_backtest(rows, BacktestConfig(start=date(2026, 9, 1), end=date(2026, 12, 31), capital=10_000, fee_model="Kalshi",
                                             slippage=0.0, prob_mode=PROB_MODES[2]))
    assert len(full.trades) > len(r.trades) and abs(full.trades.p_used.iloc[0] - 0.50) < 1e-9     # hindsight mode bets from day one at 15c + 35c edge


def test_backtest_skeptical_start_needs_evidence():
    # no edge until the history itself beats the price: nothing at the start, first bet only once p_hat clears price + fee + min edge
    rows = _rows([0, 0, 0, 0, 0] + [1] * 60)
    cfg = BacktestConfig(start=date(2026, 9, 1), end=date(2026, 12, 31), capital=10_000, fee_model="Kalshi", slippage=0.0)
    r = run_backtest(rows, cfg)
    assert len(r.trades) >= 1
    first_day = (r.trades.entry_ts.iloc[0] - datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp()) / 86400
    assert first_day > 5 and r.trades.p_used.iloc[0] > 0.15 + 0.005


def test_signal_stats_formulas_and_no_lookahead():
    sig = pd.DataFrame(dict(leg=["DOG"] * 10, hour=[22] * 5 + [3] * 5, close_ts=[1000 * (i + 1) for i in range(10)], open_ts=[1000 * (i + 1) - 900 for i in range(10)],
                            win=[1, 0, 0, 0, 1, 0, 0, 0, 0, 0], cost=[0.2] * 10))
    # edge basis, overall: d = sum(win-cost)/(n+n0); at entry 10_500 only the first 10 are settled
    st = SignalStats.from_signals(sig, mode="overall", prior_n=20, basis="edge")
    p, n, _ = st.p("DOG", 22, 0.10, 10_500)
    assert n == 10 and abs(p - (0.10 + (2 - 2.0) / 30)) < 1e-12                  # 2 wins - 10*0.2 = 0 -> p = cost (no edge)
    p, n, _ = st.p("DOG", 22, 0.10, 3_500)                                         # only 3 settled: wins=1 -> E = 1-0.6 = 0.4 -> d = 0.4/23
    assert n == 3 and abs(p - (0.10 + 0.4 / 23)) < 1e-12
    st0 = SignalStats.from_signals(sig, mode="overall", prior_n=20, basis="win_rate")
    p, n, _ = st0.p("DOG", 22, 0.10, 3_500)                                        # (wins + c*n0)/(n + n0) = (1 + 2)/23
    assert abs(p - 3 / 23) < 1e-12
    # by-hour shrink: hour 22 has 2 wins in 5 (E = 2-1 = 1), leg level d0 = 0 -> d1 = (1 + k*0)/(5+k)
    sh = SignalStats.from_signals(sig, mode="by_hour", hour_k=10, prior_n=20, basis="edge")
    p, n, nh = sh.p("DOG", 22, 0.20, 10_500)
    assert nh == 5 and abs(p - (0.20 + 1.0 / 15)) < 1e-12
    # lookback excludes old signals
    sl = SignalStats.from_signals(sig, mode="overall", prior_n=20, basis="edge", lookback_days=1)
    assert sl.p("DOG", 22, 0.2, 10_500)[1] == 10 and sl.count(("DOG", None), 10_500 + 3 * 86400)[0] == 0


def test_evaluate_quote_all_legs():
    p = StrategyParams(legs=("DOG", "FAV"), fav_min=0.85, fav_max=0.97)
    s = evaluate_quote_all(0.10, 0.11, 12, 22, p)            # YES 0.10/0.11: dog = YES @ .11 ; favourite = NO @ 1-0.10 = .90
    assert {x.leg: (x.side, round(x.cost, 2)) for x in s} == {"DOG": ("YES", 0.11), "FAV": ("NO", 0.90)}
    s = evaluate_quote_all(0.89, 0.90, 12, 22, p)            # YES is the favourite @ .90 ; dog = NO @ 1-.89 = .11
    assert {x.leg: (x.side, round(x.cost, 2)) for x in s} == {"DOG": ("NO", 0.11), "FAV": ("YES", 0.90)}
    assert evaluate_quote_all(0.50, 0.51, 12, 22, p) == []
    assert fmt_int_list((0, 1, 2, 3, 22, 23)) == "0-3,22-23" and fmt_int_list((22,)) == "22"


def test_extract_signals_two_legs_and_one_trade_per_window():
    rows = pd.DataFrame([dict(ticker="A", open_ts=int(datetime(2026, 9, 1, 22, tzinfo=timezone.utc).timestamp()), minute=12, bid=0.10, ask=0.11, yes=1),
                         dict(ticker="B", open_ts=int(datetime(2026, 9, 1, 23, tzinfo=timezone.utc).timestamp()), minute=12, bid=0.10, ask=0.11, yes=0)])
    from trader.data import extract_signals
    s = extract_signals(rows, StrategyParams(legs=("DOG", "FAV")))
    assert len(s) == 4 and set(s.leg) == {"DOG", "FAV"}
    a = s[s.ticker == "A"].set_index("leg")
    assert a.loc["DOG", "win"] == 1 and a.loc["FAV", "win"] == 0                   # YES won: the YES dog wins, the NO favourite loses
    cfg = BacktestConfig(start=date(2026, 9, 1), end=date(2026, 9, 2), capital=10_000, fee_model="Kalshi", slippage=0.0, prob_mode=PROB_MODES[1],
                         fixed_p=0.95, params=StrategyParams(legs=("DOG", "FAV")), min_edge=0.0)
    r = run_backtest(rows, cfg)
    assert r.trades.market.is_unique and len(r.trades) == 2                          # one trade per window, never both legs
    assert (r.trades.leg == "DOG").all() or (r.trades.leg == "FAV").all()


def test_signal_stats_halflife_formulas_lookahead_and_order():
    from trader.probmodel import EPOCH0
    t0 = EPOCH0 + 100 * 86400
    ages = [1, 3, 8, 20, 40]
    wins = [1, 0, 1, 0, 1]
    costs = [0.2, 0.25, 0.2, 0.3, 0.2]
    sig = pd.DataFrame(dict(leg="DOG", hour=22, close_ts=[int(t0 - a * 86400) for a in ages], open_ts=[int(t0 - a * 86400) - 900 for a in ages], win=wins, cost=costs))
    for hl in (7.0, 14.0):
        w = np.array([0.5 ** (a / hl) for a in ages])
        e = np.array([x - c for x, c in zip(wins, costs)])
        s1 = SignalStats.from_signals(sig, mode="overall", prior_n=10, halflife_days=hl, basis="edge")
        d = (w * e).sum() / (w.sum() + 10)
        assert abs(s1.p("DOG", 22, 0.25, t0)[0] - (0.25 + d)) < 1e-9
        s2 = SignalStats.from_signals(sig, mode="by_hour", hour_k=30, prior_n=10, halflife_days=hl, basis="edge")
        assert abs(s2.p("DOG", 22, 0.25, t0)[0] - (0.25 + ((w * e).sum() + 30 * d) / (w.sum() + 30))) < 1e-9
        s3 = SignalStats.from_signals(sig, mode="overall", prior_n=10, halflife_days=hl, basis="win_rate")
        assert abs(s3.p("DOG", 22, 0.25, t0)[0] - ((w * np.array(wins)).sum() + 0.25 * 10) / (w.sum() + 10)) < 1e-9
    a = SignalStats(mode="overall", prior_n=10, halflife_days=10)          # out-of-order inserts and a signal that settles AFTER the entry
    a.add("DOG", 22, int(t0 - 5 * 86400), 1, 0.2); a.add("DOG", 22, int(t0 - 9 * 86400), 0, 0.2); a.add("DOG", 22, int(t0 + 86400), 1, 0.2)
    b = SignalStats(mode="overall", prior_n=10, halflife_days=10)
    b.add("DOG", 22, int(t0 - 9 * 86400), 0, 0.2); b.add("DOG", 22, int(t0 - 5 * 86400), 1, 0.2); b.add("DOG", 22, int(t0 + 86400), 1, 0.2)
    assert abs(a.p("DOG", 22, 0.2, t0)[0] - b.p("DOG", 22, 0.2, t0)[0]) < 1e-12 and a.p("DOG", 22, 0.2, t0)[1] == 2
    try:
        SignalStats(halflife_days=1)
        assert False
    except ValueError:
        pass


def _parity_rows(outcomes, windows_per_day=4):
    """Several windows per UTC day (22:00, 22:15, ...) so the daily-loss pause matters; same quotes as the engine tests (YES bid .14 / ask .15 at minute 12)."""
    recs = []
    day0 = int(datetime(2026, 10, 9, 22, 0, tzinfo=timezone.utc).timestamp())
    for i, y in enumerate(outcomes):
        d, w = divmod(i, windows_per_day)
        o = day0 + d * 86400 + w * 900
        recs.append(dict(ticker=f"T{o}", open_ts=o, minute=12, bid=0.14, ask=0.15, yes=int(y)))
    return pd.DataFrame(recs)


def _drive_live(outcomes, cfg, windows_per_day=4):
    eng, clock, _ = _engine(cfg=cfg)
    day0 = datetime(2026, 10, 9, 22, 0, tzinfo=timezone.utc).timestamp()
    for i, y in enumerate(outcomes):
        d, w = divmod(i, windows_per_day)
        w0 = day0 + d * 86400 + w * 900
        eng.quotes.res = "yes" if y else "no"
        clock.t = w0 + 12 * 60 + 2
        eng.tick()
        clock.t = w0 + 900 + 25
        eng.tick()
    return eng


def test_live_engine_and_backtest_produce_identical_trades():
    """The same rules, the same data, the same settings -> the same trades and equity, including the risk limits and the drawdown throttle."""
    outcomes = [0, 0, 1, 1,  0, 0, 0, 1,  0, 0, 0, 0,  1, 1, 0, 1,  0, 0, 1, 0,  0, 0, 0, 0,  1, 0, 1, 1]
    common = dict(capital=1000.0, kelly_mult=0.25, max_frac=0.5, min_edge=0.04, max_contracts=2000)
    # LiveConfig has the daily-loss / drawdown limits ON by default and BacktestConfig has them OFF, so every case states them explicitly for both (the GUI passes one shared value to both); same for the drawdown throttle, which is ON by default only in LiveConfig
    for extra in (dict(daily_loss_limit=0.0, max_drawdown_stop=0.0, dd_throttle=False), dict(daily_loss_limit=0.05, max_drawdown_stop=0.25, dd_throttle=False),
                  dict(daily_loss_limit=0.05, max_drawdown_stop=0.12, dd_throttle=True)):
        live_extra = {k: ((v if v else 0.999) if k != 'dd_throttle' else v) for k, v in extra.items()}      # 0.999 = effectively off for the live engine
        lcfg = LiveConfig(prob_source="manual", manual_p=0.40, paper_slippage=0.02, **common, **live_extra)
        eng = _drive_live(outcomes, lcfg)
        bcfg = BacktestConfig(start=date(2026, 10, 9), end=date(2026, 10, 25), params=StrategyParams(cost_min=0.05, cost_max=0.30), prob_mode=PROB_MODES[1], fixed_p=0.40,
                              slippage=0.02, **common, **extra)
        bt = run_backtest(_parity_rows(outcomes), bcfg).trades
        lv = pd.DataFrame(eng.trades)
        assert len(bt) == len(lv) and len(bt) >= 2, (extra, len(bt), len(lv))                  # (the strictest case halts after 2 trades on BOTH sides)
        assert (bt.contracts.values == lv.contracts.values).all(), (extra, bt.contracts.values, lv.contracts.values)
        assert np.allclose(bt.pnl.values, lv.pnl.values) and np.allclose(bt.equity_after.values, lv.equity_after.values)
        assert np.allclose(bt.entry.values, lv.entry.values)
        # and the metrics computed from the live session's trades are the backtest's metrics (same code, same numbers)
        from trader.live import trades_frame
        from trader.metrics import evaluate_trades
        _eq, _dd, m_live = evaluate_trades(trades_frame(eng.trades), 1000.0, int(datetime(2026, 10, 9, tzinfo=timezone.utc).timestamp()), int(datetime(2026, 10, 25, tzinfo=timezone.utc).timestamp()))
        m_bt = run_backtest(_parity_rows(outcomes), bcfg).metrics
        for k in ("trades", "win_rate", "total_return", "final_equity", "max_drawdown", "profit_factor", "max_consec_losses"):
            assert abs(float(m_live[k]) - float(m_bt[k])) < 1e-9, (k, m_live[k], m_bt[k])
    # the limits really did something in the strict case
    assert len(bt) < len(run_backtest(_parity_rows(outcomes), BacktestConfig(start=date(2026, 10, 9), end=date(2026, 10, 25), params=StrategyParams(cost_min=0.05, cost_max=0.30),
                                                                              prob_mode=PROB_MODES[1], fixed_p=0.40, slippage=0.02, **common)).trades)


def test_backtest_risk_limits_and_min_history():
    outcomes = [0, 0, 0, 0, 0, 0, 1, 1]                                  # 2 windows per day: day 1 is lost entirely
    kw = dict(start=date(2026, 10, 9), end=date(2026, 10, 25), params=StrategyParams(cost_min=0.05, cost_max=0.30), prob_mode=PROB_MODES[1], fixed_p=0.40, kelly_mult=0.25, max_frac=0.5,
              min_edge=0.04, dd_throttle=False, max_contracts=10**6)
    off = dict(daily_loss_limit=0.0, max_drawdown_stop=0.0, min_history=0)                 # the defaults are the live engine's rules (ON); the 'free' baseline switches them off explicitly
    rows = _parity_rows(outcomes, windows_per_day=2)
    free = run_backtest(rows, BacktestConfig(**kw, **off)).trades
    capped = run_backtest(rows, BacktestConfig(**{**kw, **off, "daily_loss_limit": 0.05}))
    assert len(capped.trades) < len(free) and capped.metrics["windows_paused"] > 0          # day 1: first loss pauses the second window
    stopped = run_backtest(rows, BacktestConfig(**{**kw, **off, "max_drawdown_stop": 0.10}))
    assert len(stopped.trades) < len(free) and stopped.metrics["windows_halted"] > 0
    # min_history only gates the expanding-probability mode (the live engine's MIN_HISTORY rule)
    exp = dict(kw, prob_mode=PROB_MODES[0], daily_loss_limit=0.0, max_drawdown_stop=0.0)
    a = run_backtest(rows, BacktestConfig(**exp, min_history=0))
    b = run_backtest(rows, BacktestConfig(**exp, min_history=30))
    assert len(b.trades) <= len(a.trades)
    assert len(run_backtest(rows, BacktestConfig(**{**kw, **off, "min_history": 30})).trades) == len(free)      # fixed p: unaffected
    d = BacktestConfig(start=date(2026, 1, 1), end=date(2026, 1, 2))
    assert d.daily_loss_limit == 0.10 and d.max_drawdown_stop == 0.25 and d.min_history == 30 and d.dd_throttle and d.max_contracts == 2000 and d.fee_model == "Kalshi"      # the defaults ARE the live engine's


def test_drawdown_throttle_function_backtest_and_live():
    from trader.strategy import drawdown_throttle
    assert drawdown_throttle(0.0) == 1.0 and drawdown_throttle(0.079) == 1.0 and drawdown_throttle(0.08) == 0.5 and drawdown_throttle(0.149) == 0.5 and drawdown_throttle(0.15) == 0.25
    # backtest: after a run of losses the throttled run stakes less per trade than the unthrottled one
    rows = _rows([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    kw = dict(start=date(2026, 9, 1), end=date(2026, 9, 20), params=StrategyParams(cost_min=0.05, cost_max=0.30), kelly_mult=1.0, max_frac=0.5, min_edge=0.0,
              prob_mode=PROB_MODES[1], fixed_p=0.40, max_contracts=10**6, daily_loss_limit=0.0, max_drawdown_stop=0.0, min_history=0)
    a = run_backtest(rows, BacktestConfig(**kw, dd_throttle=False)).trades
    b = run_backtest(rows, BacktestConfig(**kw, dd_throttle=True)).trades
    assert len(a) == len(b) and (a.contracts.values[:3] == b.contracts.values[:3]).all()          # identical until the drawdown reaches -8%
    assert b.contracts.values[3] < 0.4 * a.contracts.values[3]                                   # first trade after the drawdown is sized down
    assert b.equity_after.values[-1] > 5 * a.equity_after.values[-1]                             # and the bankroll survives the losing streak
    # live engine: same equity ratio, deeper drawdown -> smaller position
    def qty(equity, peak, throttle):
        eng, clock, _ = _engine(cfg=LiveConfig(capital=1000.0, max_frac=0.9, dd_throttle=throttle), p=0.30)
        eng.equity, eng.peak = equity, peak
        clock.t += 12 * 60 + 2
        eng.tick()
        return next(iter(eng._positions.values())).qty
    q0 = qty(1000.0, 1000.0, True)
    assert qty(900.0, 1000.0, True) < 0.6 * q0 and qty(900.0, 1000.0, False) > 0.8 * q0 and qty(800.0, 1000.0, True) < 0.3 * q0


def test_drawdown_throttle_also_scales_a_binding_stake_cap():
    """With a binding cap (e.g. 1.0x Kelly / 5%) scaling only the multiplier left 'half size' a no-op: the throttle must shrink the cap too."""
    import trader.strategy as S
    assert S.throttled_sizing(1.0, 0.05, 1000.0, 1000.0, True) == (1.0, 0.05)
    assert S.throttled_sizing(1.0, 0.05, 900.0, 1000.0, True) == (0.5, 0.025) and S.throttled_sizing(1.0, 0.05, 800.0, 1000.0, True) == (0.25, 0.0125)
    assert S.throttled_sizing(1.0, 0.05, 800.0, 1000.0, False) == (1.0, 0.05)
    S.THROTTLE_SCALES_CAP = False
    try:
        assert S.throttled_sizing(1.0, 0.05, 900.0, 1000.0, True) == (0.5, 0.05)                      # the old behaviour, kept switchable for comparisons
    finally:
        S.THROTTLE_SCALES_CAP = True
    def qty(equity, peak, throttle):
        eng, clock, _ = _engine(cfg=LiveConfig(capital=1000.0, kelly_mult=1.0, max_frac=0.05, max_contracts=10**6, dd_throttle=throttle), p=0.45)
        eng.equity, eng.peak = equity, peak
        clock.t += 12 * 60 + 2
        eng.tick()
        return next(iter(eng._positions.values())).qty
    q_full = qty(1000.0, 1000.0, True)
    assert q_full * 0.17 <= 0.05 * 1000.0 + 0.5 and qty(900.0, 1000.0, False) > 0.85 * q_full             # the 5% cap binds, so the unthrottled stake stays at the cap
    assert qty(900.0, 1000.0, True) < 0.55 * q_full and qty(800.0, 1000.0, True) < 0.3 * q_full           # throttled: about half / a quarter of the stake


def test_core_hours_use_smaller_shrinkage():
    n = 30
    rows = []
    for hr in (5, 22):                                         # identical hot evidence in a core hour (22) and a non-core hour (5)
        for i in range(n):
            rows.append(dict(leg="DOG", hour=hr, close_ts=1_790_000_000 + i * 3600, open_ts=1_790_000_000 + i * 3600 - 900, win=int(i % 2 == 0), cost=0.2))
    for i in range(300):                                       # plus a big pool of break-even evidence in other hours (the overall estimate the hours shrink toward)
        rows.append(dict(leg="DOG", hour=11, close_ts=1_790_000_000 + i * 600, open_ts=1_790_000_000 + i * 600 - 900, win=int(i % 5 == 0), cost=0.2))
    sig = pd.DataFrame(rows)
    now = 1_790_000_000 + 400 * 3600
    plain = SignalStats.from_signals(sig, mode="by_hour", hour_k=100, prior_n=20, basis="edge")
    core = SignalStats.from_signals(sig, mode="by_hour", hour_k=100, prior_n=20, basis="edge", core_hours=(22,), core_k=10)
    assert abs(plain.p("DOG", 22, 0.2, now)[0] - plain.p("DOG", 5, 0.2, now)[0]) < 1e-12          # same evidence, same shrinkage
    p22, p5 = core.p("DOG", 22, 0.2, now)[0], core.p("DOG", 5, 0.2, now)[0]
    assert p22 > p5 + 0.05 and abs(p5 - plain.p("DOG", 5, 0.2, now)[0]) < 1e-12                  # the core hour trusts its own evidence more; the others are untouched
    off = SignalStats.from_signals(sig, mode="by_hour", hour_k=100, prior_n=20, basis="edge", core_hours=(22,), core_k=0)
    assert abs(off.p("DOG", 22, 0.2, now)[0] - plain.p("DOG", 22, 0.2, now)[0]) < 1e-12           # core_k = 0 disables it


def test_decaying_memory_stops_betting_sooner_when_the_edge_disappears():
    # 60 hot days (win 55% at 15c), then 40 dead days (win 12% at 15c): count the bets placed in the dead stretch
    rng = np.random.default_rng(5)
    outcomes = list((rng.random(60) < 0.55).astype(int)) + list((rng.random(40) < 0.12).astype(int))
    rows = _rows(outcomes)
    start, end = date(2026, 9, 1), date(2026, 12, 9)
    kw = dict(start=start, end=end, params=StrategyParams(cost_min=0.05, cost_max=0.30), kelly_mult=1.0, max_frac=0.05, min_edge=0.02, min_history=0, daily_loss_limit=0.0, max_drawdown_stop=0.0, dd_throttle=False)
    dead_from = datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp()
    def bets_in_dead(cfg_kw):
        t = run_backtest(rows, BacktestConfig(**kw, **cfg_kw)).trades
        return int((t.entry_ts >= dead_from).sum()) if len(t) else 0
    window = bets_in_dead(dict(lookback_days=45))
    decay = bets_in_dead(dict(halflife_days=7.0))
    assert decay < window and window >= 5, (window, decay)                    # the hard window keeps betting on stale evidence, the decay lets go
    # and it still learns a regime quickly: first bet within the first hot weeks
    t = run_backtest(rows, BacktestConfig(**kw, halflife_days=7.0)).trades
    assert len(t) and t.entry_ts.min() < datetime(2026, 9, 25, tzinfo=timezone.utc).timestamp()


def test_radar_finds_the_hot_hour_and_flags_a_move():
    from trader import radar
    rng = np.random.default_rng(3)
    recs = []
    for d in range(60):
        for hr in range(24):
            for q in range(4):
                o = int(datetime(2026, 8, 1, tzinfo=timezone.utc).timestamp()) + d * 86400 + hr * 3600 + q * 900
                hot = (hr == 5) and d >= 30                                                   # hour 5 turns hot half-way
                recs.append(dict(ticker=f"T{o}", open_ts=o, minute=12, bid=0.19, ask=0.20, yes=int(rng.random() < (0.55 if hot else 0.18))))
    rows = pd.DataFrame(recs)
    now = float(rows.open_ts.max()) + 3600
    df = radar.hour_radar(rows, now, active_hours=(22, 23))
    assert len(df) == 25 and df.iloc[-1].hour == "ALL"
    assert df[df.hour == 5].iloc[0].status == "TRADE" and df[df.hour == 5].iloc[0].est_edge > 0.04
    assert (df[(df.hour != 5) & (df.hour != "ALL")].status == "cold").mean() > 0.9
    state, txt = radar.summarize(df, (22, 23))
    assert state == "moved" and "5" in txt
    state, _ = radar.summarize(df, (5,))
    assert state == "ok"


def test_compute_metrics_known_drawdown():
    t = pd.DataFrame(dict(settle_ts=[1000, 2000, 3000, 4000], entry_ts=[900, 1900, 2900, 3900], pnl=[100., -200., -100., 400.],
                          stake=[500., 500., 500., 500.], fee=[1., 1., 1., 1.], equity_after=[1100., 900., 800., 1200.],
                          f_full=[.1] * 4, f_used=[.05] * 4))
    m = compute_metrics(t, 1000., 0, 86400 * 5)
    assert abs(m["max_drawdown"] - (800 - 1100) / 1100) < 1e-12 and m["max_consec_losses"] == 2 and m["max_consec_wins"] == 1
    assert m["win_rate"] == 0.5 and abs(m["profit_factor"] - 500 / 300) < 1e-12 and m["best_trade"] == 400 and m["worst_trade"] == -200


# ── live engine ───────────────────────────────────────────────────────────────
class FakeQuotes:
    def __init__(self, bid=0.14, ask=0.15, result="no"):
        self.bid, self.ask, self.res = bid, ask, result
    def window_ticker(self, open_ts): return f"KXBTC15M-{open_ts}"
    def quote(self, ticker): return (self.bid, self.ask)
    def result(self, ticker): return self.res


class Clock:
    def __init__(self, t): self.t = t
    def __call__(self): return self.t


def _stats(p=0.30, n=200, cost=0.15, basis="edge"):
    """n historical DOG signals at ``cost`` with a win rate of p (all settled long ago)."""
    wins = (np.arange(n) < int(p * n)).astype(int)
    sig = pd.DataFrame(dict(leg="DOG", hour=22, close_ts=np.arange(n) * 900 + 900, open_ts=np.arange(n) * 900, win=wins, cost=cost))
    return SignalStats.from_signals(sig, mode="overall", basis=basis)


def _engine(cfg=None, quotes=None, broker=None, p=0.30, n=200):
    clock = Clock(datetime(2026, 10, 9, 22, 0, tzinfo=timezone.utc).timestamp())
    cfg = cfg or LiveConfig(capital=1000.0)
    events = []
    eng = LiveEngine(cfg, quotes or FakeQuotes(), broker or PaperBroker(cfg.fee_model, cfg.paper_slippage), _stats(p, n),
                     lambda k, d: events.append((k, d)), clock=clock)
    return eng, clock, events


def test_live_engine_trade_cycle():
    eng, clock, events = _engine()
    w0 = clock.t
    clock.t = w0 + 12 * 60 + 2                                           # 22:12:02 -> decision minute 12
    eng.tick()
    assert len(eng._positions) == 1
    pos = next(iter(eng._positions.values()))
    assert pos.side == "YES" and pos.qty >= 1 and abs(pos.price - 0.17) < 1e-9          # quote .15 + 2c paper slippage
    assert pos.qty * pos.price + pos.fee <= 0.02 * 1000 + 1e-6                             # 2% cap respected
    clock.t = w0 + 13 * 60 + 2
    eng.tick()
    assert len(eng._positions) == 1                                                       # one trade per window
    clock.t = w0 + 900 + 25
    eng.quotes.res = "yes"                                                                # underdog YES wins
    eng.tick()
    assert len(eng.trades) == 1 and eng.trades[0]["outcome"] == "WIN" and eng.equity > 1000
    expected = pos.qty * 1.0 - pos.qty * pos.price - pos.fee
    assert abs(eng.equity - (1000 + expected)) < 1e-9
    assert any(k == "trade_close" for k, _ in events)


def test_live_engine_loss_and_hours_gate():
    eng, clock, _ = _engine()
    w0 = clock.t
    clock.t = w0 - 3600 + 12 * 60 + 2                                                     # 21:12 -> outside strategy hours
    eng.tick()
    assert not eng._positions and not eng._shadow
    clock.t = w0 + 12 * 60 + 2
    eng.tick()
    clock.t = w0 + 900 + 25
    eng.quotes.res = "no"
    eng.tick()
    assert eng.trades[0]["outcome"] == "LOSS" and eng.equity < 1000


def test_live_engine_kelly_zero_and_min_history():
    # historical p (15%) below the all-in price -> no bet, but the signal is still tracked for the probability estimate
    eng, clock, _ = _engine(p=0.15)
    clock.t += 12 * 60 + 2
    eng.tick()
    assert not eng._positions and len(eng._shadow) == 1
    eng2, clock2, _ = _engine(n=10)                                                         # too little history
    clock2.t += 12 * 60 + 2
    eng2.tick()
    assert not eng2._positions


def test_live_engine_halts_on_broker_error_and_kill():
    class BadBroker:
        mode = "live"
        def buying_power(self): return 500.0
        def buy(self, *a): raise BrokerError("state unknown")
        def cancel_all(self): self.cancelled = True
    b = BadBroker()
    eng, clock, events = _engine(cfg=LiveConfig(capital=1000.0, mode="live"), broker=b)
    clock.t += 12 * 60 + 2
    eng.tick()
    assert eng.halted and "state unknown" in eng.halt_reason
    eng.kill()
    assert getattr(b, "cancelled", False)


def test_live_engine_caps_stake_to_buying_power():
    class Broker:
        mode = "live"
        sizes = []
        def buying_power(self): return 100.0
        def buy(self, ticker, side, n, mx, q):
            self.sizes.append(n); return Fill(n, q, 0.0, "x")
        def cancel_all(self): pass
    b = Broker()
    eng, clock, _ = _engine(cfg=LiveConfig(capital=10_000.0, mode="live"), broker=b)
    clock.t += 12 * 60 + 2
    eng.tick()
    assert b.sizes and b.sizes[0] * 0.18 <= 0.02 * 100.0 + 0.5          # sized off $100 buying power, not $10,000 equity


def test_live_engine_drawdown_stop():
    eng, clock, _ = _engine(cfg=LiveConfig(capital=1000.0, max_drawdown_stop=0.1))
    eng.equity = 880.0
    eng.tick()
    assert eng.halted and "drawdown" in eng.halt_reason


# ── Auto mode (trader.adaptive) ───────────────────────────────────────────────
def _hot_flat_signals(phases, per_day=8, cost=0.20, seed=7):
    """Synthetic DOG signals in hour 22: ``phases`` = [(days, win_rate), ...]; returns (signals, days)."""
    from trader.probmodel import EPOCH0
    rng = np.random.default_rng(seed)
    d0 = EPOCH0 + 200 * 86400
    rows, d = [], 0
    for days, wr in phases:
        for _ in range(days):
            for k in range(per_day):
                o = int(d0 + d * 86400 + 22 * 3600 + k * 225)
                rows.append(dict(leg="DOG", hour=22, open_ts=o, minute=12, close_ts=o + 900, win=int(rng.random() < wr), cost=cost, ticker=f"T{d}_{k}"))
            d += 1
    return pd.DataFrame(rows), d, d0


def _auto_stats(phases, auto=None):
    from trader.adaptive import AdaptiveStats, AutoConfig
    sig, nd, d0 = _hot_flat_signals(phases)
    auto = auto or AutoConfig()
    st = AdaptiveStats.from_signals(sig, auto, mode="by_hour", hour_k=100, prior_n=20, basis="edge", core_hours=(22, 23), core_k=10)
    return st, auto, d0 + nd * 86400 + 22 * 3600 + 12 * 60


def test_auto_parse_ladder_and_tiers():
    from trader.adaptive import AutoConfig, DEFAULT_LADDER, format_ladder, parse_ladder
    lad = parse_ladder("0:0.5/3, 25_000:0.35/2.5; 100000:0.25/2")
    assert lad == ((0.0, 0.5, 0.03), (25000.0, 0.35, 0.025), (100000.0, 0.25, 0.02))
    assert parse_ladder(format_ladder(DEFAULT_LADDER)) == DEFAULT_LADDER                   # round trip
    assert parse_ladder("5000:0.4/2")[0][0] == 0.0                                          # a ladder not starting at 0 is extended downwards with its first step
    for bad in ("", "abc", "0:0.5", "0:1.5/3", "0:0.5/80", "0:0/3"):
        try:
            parse_ladder(bad)
            assert False, bad
        except ValueError:
            pass
    a = AutoConfig(ladder=lad)
    assert a.tier(0.0)[0] == 0 and a.tier(24_999.0)[0] == 0 and a.tier(25_000.0)[0] == 1 and a.tier(10**9)[0] == 2
    assert a.tier(30_000.0)[1:] == (0.35, 0.025)
    assert a.regime_scale("HOT") == 1.0 and a.regime_scale("COLD") == 0.0 and a.regime_scale("??") == 0.0


def test_auto_learners_regime_labels_and_weights():
    from trader.adaptive import AutoConfig, classify_regime
    seen = {}
    for name, phases in (("flat", [(90, .20)]), ("hot5", [(85, .20), (5, .55)]), ("hot20", [(70, .20), (20, .55)]), ("dead8", [(70, .20), (30, .55), (8, .12)]), ("dead20", [(70, .20), (30, .55), (20, .12)])):
        st, auto, t = _auto_stats(phases, AutoConfig(z=0.5))             # z = 0.5 haircut: the sharper test of the labels (the default z = 0.25 is checked below)
        pr = st.predict("DOG", 22, 0.20, t)
        assert abs(sum(pr.weights) - 1.0) < 1e-9 and min(pr.weights) >= 0.9 * auto.mix_floor, (name, pr.weights)       # weights are a proper mixture with the floor respected
        assert pr.d_mix >= min(pr.d_fast, pr.d_slow) - 1e-12 and pr.d_mix <= max(pr.d_fast, pr.d_slow) + 1e-12
        assert pr.p_used < 0.20 + pr.d_mix + 1e-12                                                                      # the haircut only lowers p
        seen[name] = (pr, classify_regime(pr, 0.036, 0.04, auto))
    assert seen["flat"][1] == "COLD" and seen["hot5"][1] == "BUILDING" and seen["hot20"][1] == "HOT" and seen["dead8"][1] == "FADING"      # no edge / new regime / confirmed / just ended
    assert seen["hot20"][0].weights[0] > seen["hot20"][0].weights[-1]                                                     # right after a regime change the fast learner earns the larger weight
    assert seen["dead20"][0].d_fast < seen["dead20"][0].d_slow                                                            # the fast memory forgets first
    assert seen["flat"][0].p_used < 0.20 + 0.04                                                                           # and a flat history never clears the hurdle
    assert AutoConfig().z == 0.25
    for name, want in (("flat", "COLD"), ("hot5", "BUILDING"), ("hot20", "HOT"), ("dead20", "FADING")):                   # the default configuration labels the clear cases the same way
        st, auto, t = _auto_stats(dict(flat=[(90, .20)], hot5=[(85, .20), (5, .55)], hot20=[(70, .20), (20, .55)], dead20=[(70, .20), (30, .55), (20, .12)])[name])
        assert classify_regime(st.predict("DOG", 22, 0.20, t), 0.036, 0.04, auto) == want, name


def test_auto_decide_ladder_regime_throttle_liquidity():
    from trader.adaptive import AutoConfig, decide
    st, auto, t = _auto_stats([(70, .20), (20, .55)])
    kw = dict(fee_model="Kalshi", slippage=0.02, min_edge=0.04)
    d = decide(auto, st, "DOG", 22, 0.20, t, 1000.0, 1000.0, 1000.0, None, **kw)
    assert d.regime == "HOT" and d.tier == 0 and abs(d.mult - 0.5) < 1e-12 and abs(d.cap - 0.03) < 1e-12 and d.sizing.contracts >= 1
    assert d.max_contracts == auto.max_contracts and d.liq_cap is None                                                     # no volume reading -> only the hard ceiling
    assert d.sizing.contracts * (0.22 + 0.01) <= 0.03 * 1000.0 + 1.0                                                     # the stake cap is respected
    big = decide(auto, st, "DOG", 22, 0.20, t, 30_000.0, 30_000.0, 30_000.0, None, **kw)
    assert big.tier == 1 and abs(big.mult - 0.35) < 1e-12 and abs(big.cap - 0.025) < 1e-12                                 # bigger account -> smaller fraction
    assert big.sizing.contracts * 0.22 <= 0.025 * 30_000.0 + 1.0
    dd = decide(auto, st, "DOG", 22, 0.20, t, 900.0, 1000.0, 900.0, None, **kw)
    assert abs(dd.mult - 0.25) < 1e-12 and abs(dd.cap - 0.015) < 1e-12                                                     # -10% drawdown halves BOTH the multiplier and the cap
    off = decide(AutoConfig(throttle=False), st, "DOG", 22, 0.20, t, 900.0, 1000.0, 900.0, None, **kw)
    assert abs(off.mult - 0.5) < 1e-12
    liq = decide(AutoConfig(), st, "DOG", 22, 0.20, t, 1_000_000.0, 1_000_000.0, 1_000_000.0, 500.0, **kw)
    assert liq.liq_cap == 50 and liq.max_contracts == 50 and liq.sizing.contracts == 50                                  # 10% of last minute's 500 contracts
    nan = decide(AutoConfig(), st, "DOG", 22, 0.20, t, 1_000_000.0, 1_000_000.0, 1_000_000.0, float("nan"), **kw)
    assert nan.liq_cap is None and nan.max_contracts == 200_000
    zero = decide(AutoConfig(), st, "DOG", 22, 0.20, t, 1000.0, 1000.0, 1000.0, 0.0, **kw)
    assert zero.sizing.contracts == 0 and "liquidity" in zero.sizing.reason
    off_liq = decide(AutoConfig(participation=0.0), st, "DOG", 22, 0.20, t, 1_000_000.0, 1_000_000.0, 1_000_000.0, 500.0, **kw)
    assert off_liq.liq_cap is None and off_liq.sizing.contracts > 50
    cold, ca, ct = _auto_stats([(90, .20)])
    c = decide(ca, cold, "DOG", 22, 0.20, ct, 1000.0, 1000.0, 1000.0, 1e6, **kw)
    assert c.regime == "COLD" and c.sizing.contracts == 0 and "regime COLD" in c.sizing.reason


def _auto_rows(d0, d1, base, rng, p_win=0.30):
    recs = []
    for d in range(d0, d1):
        for w in range(4):
            o = base + d * 86400 + w * 900
            recs.append(dict(ticker=f"T{o}", open_ts=o, minute=12, bid=0.14, ask=0.15, yes=int(rng.random() < p_win), vol=400_000.0 if (d + w) % 2 == 0 else 4_000.0))
    return pd.DataFrame(recs)


class VolQuotes(FakeQuotes):
    """FakeQuotes that also reports the previous minute's traded volume (what Kalshi's candlesticks give)."""
    def __init__(self, vols):
        super().__init__()
        self.vols, self.res_by = vols, {}
    def minute_volume(self, ticker, open_ts, minute): return self.vols[open_ts]
    def result(self, ticker): return self.res_by[int(ticker.rsplit("-", 1)[1])]


def test_auto_live_engine_and_backtest_produce_identical_trades():
    """Auto mode: the live engine and the backtest take the SAME decisions (regime, ladder tier, throttle, liquidity cap) on the same data."""
    from trader.adaptive import AutoConfig
    from trader.live import build_stats, trades_frame
    from trader.metrics import evaluate_trades
    base = int(datetime(2026, 9, 1, 22, 0, tzinfo=timezone.utc).timestamp())
    rng = np.random.default_rng(21)
    rows = _auto_rows(0, 46, base, rng)                                                  # 40 history days + 6 "live" days
    hist, fut = rows[rows.open_ts < base + 40 * 86400], rows[rows.open_ts >= base + 40 * 86400]
    auto = AutoConfig(ladder=((0.0, 0.5, 0.03), (150_000.0, 0.3, 0.02)), z=0.25, participation=0.10)
    common = dict(capital=140_000.0, fee_model="Kalshi", min_edge=0.04, daily_loss_limit=0.10, max_drawdown_stop=0.25, p_model="by_hour", hour_k=40.0, prior_n=20.0)
    params = StrategyParams(cost_min=0.05, cost_max=0.30)
    lcfg = LiveConfig(params=params, paper_slippage=0.02, auto=auto, **common)
    clock = Clock(float(base + 40 * 86400))
    quotes = VolQuotes({int(o): float(v) for o, v in zip(rows.open_ts, rows.vol)})
    eng = LiveEngine(lcfg, quotes, PaperBroker(lcfg.fee_model, lcfg.paper_slippage), build_stats(hist, lcfg), lambda k, d: None, clock=clock)
    assert eng.auto_on
    for o, y in zip(fut.open_ts, fut.yes):
        quotes.res_by[int(o)] = "yes" if y else "no"
        clock.t = float(o) + 12 * 60
        eng.tick()
        clock.t = float(o) + 900 + 25
        eng.tick()
    bcfg = BacktestConfig(start=date(2026, 10, 11), end=date(2026, 10, 25), params=params, slippage=0.02, auto=auto, min_history=30, **common)
    bt = run_backtest(rows, bcfg).trades
    lv = pd.DataFrame(eng.trades)
    assert len(bt) == len(lv) and len(bt) >= 8, (len(bt), len(lv))
    assert (bt.contracts.values == lv.contracts.values).all(), (bt.contracts.values, lv.contracts.values)
    assert np.allclose(bt.pnl.values, lv.pnl.values) and np.allclose(bt.equity_after.values, lv.equity_after.values) and np.allclose(bt.p_used.values, lv.p_used.values, atol=1e-9)
    assert (bt.regime.values == lv.regime.values).all() and np.allclose(bt.mult_used.values, lv.mult_used.values) and np.allclose(bt.cap_used.values, lv.cap_used.values)
    # the scenario really exercised the Auto features: the liquidity cap bound on thin-volume windows, and the ladder / regime / throttle changed the effective Kelly fraction
    thin = bt[bt.entry_ts.isin([int(o) + 12 * 60 for o, v in zip(rows.open_ts, rows.vol) if v < 10_000])]
    assert len(thin) >= 2 and (thin.contracts <= 400).all() and bt.contracts.max() > 5_000
    assert bt.mult_used.nunique() >= 2 and set(bt.regime) <= {"HOT", "BUILDING", "FADING"}
    _eq, _dd, m_live = evaluate_trades(trades_frame(eng.trades), 140_000.0, int(datetime(2026, 10, 11, tzinfo=timezone.utc).timestamp()), int(datetime(2026, 10, 25, tzinfo=timezone.utc).timestamp()))
    m_bt = run_backtest(rows, bcfg).metrics
    for k in ("trades", "win_rate", "total_return", "final_equity", "max_drawdown", "profit_factor"):
        assert abs(float(m_live[k]) - float(m_bt[k])) < 1e-9, (k, m_live[k], m_bt[k])
    snap = eng.auto_snapshot(clock.t)
    assert snap and set(snap) >= {"regime", "mult", "cap", "weights", "live_hours", "max_contracts"} and abs(sum(snap["weights"]) - 1.0) < 1e-9


def test_auto_with_neutral_settings_reduces_to_the_fixed_engine():
    """One learner, no evidence haircut, one ladder step, no regime scaling, no liquidity cap, no throttle  ==  the fixed-parameter engine with the same memory and size."""
    from trader.adaptive import AutoConfig
    base = int(datetime(2026, 9, 1, 22, 0, tzinfo=timezone.utc).timestamp())
    rows = _auto_rows(0, 60, base, np.random.default_rng(5), p_win=0.28)
    flat = tuple((r, 1.0) for r in ("HOT", "BUILDING", "FADING", "COLD"))
    auto = AutoConfig(half_lives=(14.0,), z=0.0, regime_mult=flat, ladder=((0.0, 0.25, 0.02),), participation=0.0, max_contracts=2000, throttle=False)
    kw = dict(start=date(2026, 10, 1), end=date(2026, 10, 25), params=StrategyParams(cost_min=0.05, cost_max=0.30), capital=10_000.0, fee_model="Kalshi", slippage=0.02, min_edge=0.04,
              min_history=30, p_model="by_hour", hour_k=40.0, dd_throttle=False, daily_loss_limit=0.10, max_drawdown_stop=0.25)
    fixed = run_backtest(rows, BacktestConfig(kelly_mult=0.25, max_frac=0.02, max_contracts=2000, halflife_days=14.0, **kw)).trades
    auto_t = run_backtest(rows, BacktestConfig(auto=auto, **kw)).trades
    assert len(fixed) == len(auto_t) and len(fixed) >= 20
    assert (fixed.contracts.values == auto_t.contracts.values).all() and np.allclose(fixed.equity_after.values, auto_t.equity_after.values)
    assert np.allclose(fixed.p_used.values, auto_t.p_used.values, atol=1e-9)
    assert (fixed.regime == "").all() and (fixed.cap_used == 0.02).all() and np.allclose(fixed.mult_used, 0.25)          # the fixed engine logs its own (constant) multiplier and cap


def test_kalshi_minute_volume_reads_the_candle_that_ended_at_the_decision_minute():
    from urllib.parse import parse_qs, urlparse
    import trader.kalshi as K

    class H(BaseHTTPRequestHandler):
        paths = []
        def log_message(self, *a): pass
        def do_GET(self):
            H.paths.append(self.path)
            u = urlparse(self.path)
            end = int(parse_qs(u.query)["end_ts"][0])
            if "/BAD/" in u.path:
                self.send_response(500); self.end_headers(); return
            cands = ([{"end_period_ts": end - 60, "volume_fp": "999.00"}, {"end_period_ts": end, "volume_fp": "170000.50"}] if "/OKT/" in u.path else [{"end_period_ts": end + 60, "volume_fp": "5"}])
            body = json.dumps({"candlesticks": cands}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    old = K.BASE
    K.BASE = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        q = K.KalshiQuotes()
        o = 1_790_000_000
        assert q.minute_volume("OKT", o, 12) == 170000.5                         # the candle whose END is open + 12 min = the last completed minute at the decision instant
        assert f"start_ts={o + 11 * 60}" in H.paths[-1] and f"end_ts={o + 12 * 60}" in H.paths[-1]
        assert q.minute_volume("NOMATCH", o, 12) is None and q.minute_volume("BAD", o, 12) is None
        n = len(H.paths)
        assert q.minute_volume("BAD", o, 12) is None and len(H.paths) == n + 1       # a single attempt: no retries on the order path
    finally:
        K.BASE = old
        srv.shutdown()
    K.BASE = "http://127.0.0.1:9"                                                  # nothing listens: None, not an exception
    try:
        assert K.KalshiQuotes().minute_volume("OKT", 1_790_000_000, 12) is None
    finally:
        K.BASE = old


def test_auto_live_volume_fallback_is_conservative():
    """When the volume read fails the engine must not size as if liquidity were unlimited: median of recent readings, else a small default."""
    from trader.adaptive import AutoConfig
    from trader.live import build_stats
    base = int(datetime(2026, 9, 1, 22, 0, tzinfo=timezone.utc).timestamp())
    rows = _auto_rows(0, 41, base, np.random.default_rng(31))
    hist = rows[rows.open_ts < base + 40 * 86400]
    wins = [base + 40 * 86400 + k * 900 for k in range(3)]

    class Q(FakeQuotes):
        def __init__(self, seq):
            super().__init__(); self.seq = list(seq); self.res = "no"
        def minute_volume(self, ticker, open_ts, minute): return self.seq.pop(0)

    auto = AutoConfig(participation=0.10, z=0.25)
    cfg = LiveConfig(capital=5_000_000.0, params=StrategyParams(cost_min=0.05, cost_max=0.30), auto=auto, p_model="by_hour", hour_k=40.0, daily_loss_limit=0.99, max_drawdown_stop=0.99)
    clock = Clock(float(wins[0]))
    quotes = Q([None, 50_000.0, None])
    eng = LiveEngine(cfg, quotes, PaperBroker(cfg.fee_model, cfg.paper_slippage), build_stats(hist, cfg), lambda k, d: None, clock=clock)
    qty = []
    for w in wins:
        clock.t = float(w) + 12 * 60
        eng.tick()
        qty.append(next(iter(eng._positions.values())).qty if eng._positions and w in eng._positions else None)
        clock.t = float(w) + 900 + 25
        eng.tick()
        eng._positions.clear()
    q0 = [t["contracts"] for t in eng.trades]
    assert len(q0) == 3 and q0[0] == int(0.10 * LiveEngine.VOLUME_FALLBACK)          # no reading ever: 10% of the small default (200), not the 200,000 ceiling
    assert q0[1] == 5000                                                           # a real reading: 10% of 50,000
    assert q0[2] == 5000                                                           # failed again: the median of the readings seen (50,000), not the default and not unlimited


def test_auto_live_without_a_volume_source_uses_only_the_hard_ceiling():
    from trader.adaptive import AutoConfig
    from trader.live import build_stats
    base = int(datetime(2026, 9, 1, 22, 0, tzinfo=timezone.utc).timestamp())
    rows = _auto_rows(0, 41, base, np.random.default_rng(31))
    hist = rows[rows.open_ts < base + 40 * 86400]
    cfg = LiveConfig(capital=5_000_000.0, params=StrategyParams(cost_min=0.05, cost_max=0.30), auto=AutoConfig(max_contracts=777, z=0.25, ladder=((0.0, 0.5, 0.03),)), p_model="by_hour", hour_k=40.0)
    clock = Clock(float(base + 40 * 86400))
    eng = LiveEngine(cfg, FakeQuotes(), PaperBroker(cfg.fee_model, cfg.paper_slippage), build_stats(hist, cfg), lambda k, d: None, clock=clock)   # FakeQuotes has no minute_volume
    clock.t += 12 * 60
    eng.tick()
    assert next(iter(eng._positions.values())).qty == 777


def test_needs_refresh_policy():
    from trader.data import REFRESH_GAP_S, REFRESH_LAG_S, needs_refresh
    now = 1_790_000_000.0
    rows = pd.DataFrame(dict(ticker=["A"], open_ts=[int(now - 3600)], minute=[12], bid=[0.1], ask=[0.11], yes=[1]))          # newest window closed 45 min ago
    assert needs_refresh(rows, now, 0.0)                                                    # stale and never refreshed
    assert not needs_refresh(rows, now, now - 60)                                           # stale, but we asked the API 1 minute ago
    assert needs_refresh(rows, now, now - REFRESH_GAP_S - 1)
    fresh = pd.DataFrame(dict(ticker=["A"], open_ts=[int(now - 900 - 60)], minute=[12], bid=[0.1], ask=[0.11], yes=[1]))      # newest window closed 1 minute ago: nothing newer can exist yet
    assert not needs_refresh(fresh, now, 0.0) and REFRESH_LAG_S >= 900
    assert needs_refresh(pd.DataFrame(columns=["ticker", "open_ts", "minute", "bid", "ask", "yes"]), now, now)             # no data at all: always


def test_collect_gapfill_adds_the_newest_windows_and_markets_merge_never_drops_rows():
    import csv
    import collect as C
    from datetime import datetime, timezone as _tz
    from pathlib import Path
    now = int(time.time())

    def iso(t):
        return datetime.fromtimestamp(t, _tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def candle(o, k):
        e = o + 60 * k
        blk = lambda v: {"open_dollars": str(v), "high_dollars": str(v), "low_dollars": str(v), "close_dollars": str(v)}
        return {"end_period_ts": e, "yes_bid": blk(0.14), "yes_ask": blk(0.15), "price": blk(0.145), "volume_fp": "1000.5"}

    hits = []
    o_ok, o_early, o_old = now - 4000, now - 3000, now - 2000

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            from urllib.parse import urlparse
            path = urlparse(self.path).path
            hits.append(path)
            if path.endswith("/candlesticks") and "/OK/" in path:
                self._send({"candlesticks": [candle(o_ok, k) for k in range(1, 16)]})
            elif path.endswith("/candlesticks") and "/EARLY/" in path:
                self._send({"candlesticks": [candle(o_early, k) for k in range(1, 10)]})       # only minutes 1-9 published so far
            elif path == "/markets":
                self._send({"markets": [dict(ticker="NEW1", open_time=iso(now - 1800), close_time=iso(now - 900), floor_strike="100", expiration_value="101", result="yes", volume_fp="5", open_interest_fp="2")], "cursor": ""})
            else:
                self._send({}, 404)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    tmpd = Path(tempfile.mkdtemp())
    old_b, old_data = C.B, C.DATA
    C.B, C.DATA = f"http://127.0.0.1:{srv.server_address[1]}", tmpd
    try:
        with open(tmpd / "markets.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, C.MARKET_FIELDS)
            w.writeheader()
            for i in range(1200):                                                    # an existing history (> 1000 rows: the historical partition is not re-read)
                w.writerow(dict(ticker=f"OLD{i}", open_ts=1_700_000_000 + i * 900, close_ts=1_700_000_900 + i * 900, strike=1, expiration_value=1, result="no", volume=1, open_interest=1))
            for t, o in (("OK", o_ok), ("EARLY", o_early), ("MISSING", o_old)):
                w.writerow(dict(ticker=t, open_ts=o, close_ts=o + 900, strike=1, expiration_value=1, result="yes", volume=1, open_interest=1))
            w.writerow(dict(ticker="JUSTCLOSED", open_ts=now - 960, close_ts=now - 60, strike=1, expiration_value=1, result="yes", volume=1, open_interest=1))     # closed < 90 s ago: not fetched yet
        with open(tmpd / "candles.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(C.CANDLE_FIELDS)
            w.writerow(["OLD0", 12, 0, .1, .1, .1, .1, .2, .2, .2, .2, .15, .15, .15, .15, 5])
        C.collect_gapfill(days=3.0)
        lines = list(csv.DictReader(open(tmpd / "candles.csv")))
        by = {}
        for r in lines:
            by.setdefault(r["ticker"], []).append(int(r["minute"]))
        assert sorted(by["OK"]) == list(range(1, 16)) and "EARLY" not in by and "MISSING" not in by and "JUSTCLOSED" not in by and by["OLD0"] == [12]        # only complete recent markets are added
        n_before = len(lines)
        C.collect_gapfill(days=3.0)                                                  # idempotent: nothing is added twice
        assert len(list(csv.DictReader(open(tmpd / "candles.csv")))) == n_before
        hits.clear()
        C.collect_markets()                                                          # incremental merge
        rows = list(csv.DictReader(open(tmpd / "markets.csv")))
        tick = {r["ticker"] for r in rows}
        assert "NEW1" in tick and {"OLD0", "OLD1199", "OK", "EARLY"} <= tick and len(rows) == 1200 + 4 + 1
        assert "/historical/markets" not in hits and "/markets" in hits              # the big historical partition is skipped when the file already has history
        assert not (tmpd / "markets.csv.tmp").exists()
    finally:
        C.B, C.DATA = old_b, old_data
        srv.shutdown()


def _ledger_engine(path, mode_broker=None, session_offset=0):
    from trader.ledger import Ledger
    clock = Clock(datetime(2026, 10, 9, 22, 0, tzinfo=timezone.utc).timestamp() + session_offset)
    cfg = LiveConfig(capital=1000.0)
    led = Ledger(path)
    eng = LiveEngine(cfg, FakeQuotes(), mode_broker or PaperBroker(cfg.fee_model, cfg.paper_slippage), _stats(0.30, 200), lambda k, d: None, clock=clock, ledger=led)
    return eng, clock, led


def test_ledger_keeps_every_trade_across_restarts_and_recovers_open_positions():
    from trader.ledger import Ledger, pooled_frame
    d = Path(tempfile.mkdtemp())
    path = d / "live" / "ledger.jsonl"
    eng, clock, led = _ledger_engine(path)
    w0 = clock.t
    clock.t = w0 + 12 * 60 + 2
    eng.tick()                                                                                  # trade 1 opens ...
    st = led.load()
    assert len(st.open) == 1 and st.settled == [] and st.open[0]["mode"] == "paper" and abs(st.open[0]["entry"] - 0.17) < 1e-9      # ... and is on disk the moment it is filled
    clock.t = w0 + 900 + 25
    eng.quotes.res = "yes"
    eng.tick()                                                                                  # trade 1 settles (WIN)
    st = led.load()
    assert len(st.settled) == 1 and st.open == [] and st.settled[0]["outcome"] == "WIN" and abs(st.settled[0]["pnl"] - eng.trades[0]["pnl"]) < 1e-9
    assert st.settled[0]["session"] == eng.session_id and st.settled[0]["recovered"] is False
    # trade 2 opens in the next window, then the app is closed before it settles
    clock.t = w0 + 900 + 12 * 60 + 2
    eng.quotes.res = "no"
    eng.tick()
    qty2, entry2, fee2 = (lambda o: (o["contracts"], o["entry"], o["fee"]))(led.load().open[0])
    assert len(eng._positions) == 1
    # ---- restart: brand-new engine / ledger objects on the same file ----
    led2 = Ledger(path)
    st2 = led2.load()
    assert len(st2.settled) == 1 and len(st2.open) == 1 and st2.open[0]["market"].endswith(str(int(w0 + 900)))
    w1 = int(w0 + 900)
    assert led2.reconcile(lambda t: "yes", now=w1 + 900 + 5) == []                               # result not posted yet (20 s settle delay): nothing is guessed
    assert led2.reconcile(lambda t: None, now=w1 + 3600) == []                                  # no result from Kalshi: stays open
    assert led2.reconcile(lambda t: "yes", now=w1 + 3600, skip_sessions=[eng.session_id]) == []  # a session that is still running settles its own positions
    rec = led2.reconcile(lambda t: "yes", now=w1 + 3600)
    assert len(rec) == 1 and rec[0]["outcome"] == "WIN" and abs(rec[0]["pnl"] - (qty2 * 1.0 - qty2 * entry2 - fee2)) < 1e-9
    assert led2.reconcile(lambda t: "yes", now=w1 + 7200) == []                                  # idempotent
    st3 = led2.load()
    assert len(st3.settled) == 2 and st3.open == [] and st3.settled[1]["recovered"] is True and st3.count("paper") == 2 and st3.count("live") == 0
    # a trade from a later session (restart) is appended to the same history
    eng3, clock3, led3 = _ledger_engine(path, session_offset=86400)
    eng3.session_id = eng.session_id + "_b"                                                      # (two sessions started within the same second in this test)
    clock3.t += 12 * 60 + 2
    eng3.tick()
    clock3.t += 900 - 12 * 60 - 2 + 25
    eng3.quotes.res = "no"
    eng3.tick()
    st4 = Ledger(path).load()
    assert len(st4.settled) == 3 and len({r["session"] for r in st4.settled}) == 2 and st4.settled[-1]["outcome"] == "LOSS"
    # pooled view: ONE equity story over all sessions (first starting bankroll + cumulative P&L), metrics through the shared evaluation
    t, cap0 = pooled_frame(st4.settled)
    assert abs(cap0 - 1000.0) < 1e-9 and len(t) == 3 and abs(t.equity_after.iloc[-1] - (1000.0 + t.pnl.sum())) < 1e-9 and {"session", "mode", "recovered"} <= set(t.columns)
    assert (t.entry_ts.values[1:] > t.entry_ts.values[:-1]).all() and np.allclose(t.bankroll_before.values[1:], t.equity_after.values[:-1])
    from trader.metrics import evaluate_trades
    _eq, _dd, m = evaluate_trades(t, cap0, int(t.entry_ts.min()) - 900, int(t.settle_ts.max()))
    assert m["trades"] == 3 and abs(m["final_equity"] - t.equity_after.iloc[-1]) < 1e-9


def test_ledger_tolerates_a_torn_line_merges_old_session_csvs_and_filters_by_mode():
    import csv as _csv
    from trader.ledger import Ledger
    from trader.live import TRADE_FIELDS
    d = Path(tempfile.mkdtemp())
    path = d / "ledger.jsonl"
    L = Ledger(path)
    t0 = 1_791_700_000                                                                           # after the old CSV's trade (2026-10-08)
    tid = L.record_open(mode="live", session="s_live", market="MKT-A", window=t0, leg="DOG", side="YES", quoted=0.15, entry=0.17, contracts=100, stake=17.3, fee=0.3, p_used=0.3, f_full=0.1, f_used=0.02,
                        bankroll_before=500.0, order_id="oid1", entry_ts=t0 + 725, regime="HOT", mult_used=0.5, cap_used=0.03)
    L.record_settle(tid, settle_ts=t0 + 925, outcome="LOSS", pnl=-17.3, equity_after=482.7)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"ev":"open","id":"torn|line')                                              # a crash in the middle of a write
        fh.write("\n\nnot json at all\n")
    st = L.load()
    assert len(st.settled) == 1 and st.bad_lines == 2 and st.settled[0]["regime"] == "HOT" and st.settled[0]["mult_used"] == 0.5
    # an old per-session CSV (written before the ledger existed): merged read-only, the ledger wins on a duplicate market
    row = dict(entry_time="2026-10-08 22:12:01", settle_time="2026-10-08 22:15:25", market="MKT-OLD", leg="DOG", side="YES", quoted=0.15, entry=0.17, contracts=50, stake=8.6, fee=0.2, outcome="WIN",
               pnl=41.2, p_used=0.3, f_full=0.1, f_used=0.02, equity_after=541.2, mode="paper", order_id="paper")
    dup = dict(row, market="MKT-A", mode="live")
    with open(d / "session_20261008_221200_paper.csv", "w", newline="") as fh:
        w = _csv.DictWriter(fh, TRADE_FIELDS)
        w.writeheader(); w.writerow(row); w.writerow(dup)
    st = L.load()
    assert [r["market"] for r in st.settled] == ["MKT-OLD", "MKT-A"] and st.legacy == 1 and st.settled[0]["session"] == "20261008_221200_paper"
    assert [r["market"] for r in L.load(mode="live").settled] == ["MKT-A"] and [r["market"] for r in L.load(mode="paper").settled] == ["MKT-OLD"]
    assert [r["market"] for r in L.load(legacy=False).settled] == ["MKT-A"]
    from trader.live import trades_frame
    t = trades_frame(L.load().settled)
    assert len(t) == 2 and list(t.market) == ["MKT-OLD", "MKT-A"] and {"session", "mode", "recovered"} <= set(t.columns)


def test_account_fills_and_the_ledger_cross_check():
    """Read-only: the account's recent fills (Kalshi GET /portfolio/fills) are compared with the ledger so a trade a crashed session never recorded is flagged."""
    from trader.ledger import Ledger
    srv, url = _mock_server()
    try:
        c = _client(url)
        MockKalshi.fills.extend([
            {"fill_id": "a", "order_id": "O-known", "ticker": "KXBTC15M-26OCT092200-00", "outcome_side": "no", "count_fp": "100.00", "created_time": "2026-10-09T22:12:03Z"},
            {"fill_id": "b", "order_id": "O-unknown", "ticker": "KXBTC15M-26OCT092300-00", "outcome_side": "no", "count_fp": "40.00", "created_time": "2026-10-09T23:12:03Z"},
            {"fill_id": "c", "order_id": "O-unknown", "ticker": "KXBTC15M-26OCT092300-00", "outcome_side": "no", "count_fp": "10.00", "created_time": "2026-10-09T23:12:04Z"},
            {"fill_id": "d", "order_id": "O-other", "ticker": "KXETH-26OCT09-T1", "outcome_side": "yes", "count_fp": "5.00", "created_time": "2026-10-09T23:12:03Z"}])
        fills = c.fills()
        assert len(fills) == 4 and [f["fill_id"] for f in c.fills(order_id="O-unknown")] == ["b", "c"]
        led = Ledger(Path(tempfile.mkdtemp()) / "ledger.jsonl")
        w22 = int(datetime(2026, 10, 9, 22, 0, tzinfo=timezone.utc).timestamp())
        led.record_open(mode="live", session="s", market="KXBTC15M-26OCT092200-00", window=w22, leg="DOG", side="NO", quoted=0.15, entry=0.17, contracts=100, stake=17.3, fee=0.3, p_used=0.3, f_full=0.1,
                        f_used=0.02, bankroll_before=500.0, order_id="O-known", entry_ts=w22 + 725)
        led.record_open(mode="paper", session="p", market="KXBTC15M-26OCT092300-00", window=w22 + 3600, leg="DOG", side="NO", quoted=0.15, entry=0.17, contracts=10, stake=1.7, fee=0.0, p_used=0.3,
                        f_full=0.1, f_used=0.02, bankroll_before=500.0, order_id="paper", entry_ts=w22 + 3600 + 725)
        miss = led.unmatched_exchange_fills(fills)
        assert [(m["order_id"], m["ticker"], m["qty"]) for m in miss] == [("O-unknown", "KXBTC15M-26OCT092300-00", 50.0)]      # explained by the ledger / not our market / a paper trade explains nothing
    finally:
        srv.shutdown()


def test_live_ledger_keeps_the_raw_fill_and_the_account_buying_power_and_marks_old_csv_trades_as_polymarket():
    import csv as _csv
    from trader.ledger import Ledger
    from trader.live import TRADE_FIELDS

    class LiveB:
        mode = "live"
        venue = "kalshi"
        def buying_power(self): return 480.0
        def buy(self, ticker, side, n, mx, q): return Fill(n, 0.17, 0.3, "oid-1", "FILLED", note="order oid-1 bid@0.20 fill_count=10")
        def cancel_all(self): pass

    d = Path(tempfile.mkdtemp())
    eng, clock, led = _ledger_engine(d / "ledger.jsonl", mode_broker=LiveB())
    w0 = clock.t
    clock.t = w0 + 12 * 60 + 2
    eng.tick()
    clock.t = w0 + 900 + 25
    eng.quotes.res = "yes"
    eng.tick()
    ev = [e for e in led.read_events()[0] if e["ev"] != "session"]
    o, st = ev[0], ev[1]
    assert o["ev"] == "open" and o["mode"] == "live" and o["venue"] == "kalshi" and "oid-1" in o["note"] and o["order_id"] == "oid-1"
    assert st["ev"] == "settle" and st["account_bp"] == 480.0 and st["outcome"] == "WIN"
    # a LIVE trade that only exists in an old per-session CSV was a Polymarket trade: the retired venue is NOT part of the history (and never explains a Kalshi fill)
    row = dict(entry_time="2026-10-09 22:27:01", settle_time="2026-10-09 22:15:25", market="KXBTC15M-OLD", leg="DOG", side="NO", quoted=0.18, entry=0.08, contracts=118, stake=10.04, fee=0.6,
               outcome="LOSS", pnl=-10.04, p_used=0.27, f_full=0.07, f_used=0.05, equity_after=489.96, mode="live", order_id="X1")
    with open(d / "session_20261009_181200_live.csv", "w", newline="") as fh:
        w = _csv.DictWriter(fh, TRADE_FIELDS)
        w.writeheader(); w.writerow(row)
    L2 = Ledger(d / "ledger.jsonl")
    assert L2.kalshi_order_ids() == {"oid-1"}
    st2 = L2.load()
    assert [r["market"] for r in st2.settled] == [o["market"]] and st2.legacy == 0 and all(r["venue"] == "kalshi" for r in st2.settled)       # only the Kalshi trade is shown
    assert [r["market"] for r in L2.load(mode="live").settled] == [o["market"]] and L2.load(mode="paper").count() == 0
    venues = {r["market"]: r["venue"] for r in L2.load(include_retired=True).settled}                                                          # still readable on request
    assert venues["KXBTC15M-OLD"] == "polymarket" and venues[o["market"]] == "kalshi"
    # a retired-venue event inside the ledger file itself (an earlier version wrote them) is hidden the same way, open or settled
    L2.record_open(mode="live", venue="polymarket", session="OLDSESS", market="KXBTC15M-PM2", window=1_780_000_000, leg="DOG", side="NO", quoted=0.2, entry=0.22, contracts=5, stake=1.2, fee=0.1, p_used=0.3, f_full=0.1,
                   f_used=0.02, bankroll_before=100.0, order_id="PM2", entry_ts=1_780_000_720)
    assert not [x for x in L2.load().open if x["market"] == "KXBTC15M-PM2"] and [x for x in L2.load(include_retired=True).open if x["market"] == "KXBTC15M-PM2"]


def test_live_quote_comes_from_the_order_book_not_the_lagging_market_object():
    from urllib.parse import urlparse
    import trader.kalshi as K
    pb = K.KalshiQuotes.parse_orderbook
    assert pb({"orderbook_fp": {"yes_dollars": [["0.30", "10"], ["0.71", "5"]], "no_dollars": [["0.20", "3"], ["0.27", "9"]]}}) == (0.71, 0.73)          # YES ask = 1 - best NO bid
    assert pb({"orderbook": {"yes": [[30, 5], [44, 2]], "no": [[50, 1]]}}) == (0.44, 0.50)                                                              # older cent format
    assert pb({"orderbook_fp": {"yes_dollars": [], "no_dollars": []}}) == (0.0, 1.0) and pb({}) is None and pb({"orderbook_fp": {"yes_dollars": [["x", "1"]]}}) is None

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            path = urlparse(self.path).path
            stale = {"market": {"yes_bid_dollars": "0.6600", "yes_ask_dollars": "0.6700"}}                      # what the market object said 10 s ago
            if path == "/markets/T1/orderbook":
                return self._send({"orderbook_fp": {"yes_dollars": [["0.71", "5"]], "no_dollars": [["0.27", "9"]]}})
            if path == "/markets/T1":
                return self._send(stale)
            if path == "/markets/T2/orderbook":
                return self._send({}, 404)
            if path == "/markets/T2":
                return self._send({"market": {"yes_bid_dollars": "0.4000", "yes_ask_dollars": "0.4100"}})
            self._send({}, 404)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    old = K.BASE
    K.BASE = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        q = K.KalshiQuotes()
        assert q.quote("T1") == (0.71, 0.73)                         # the live book, not the stale 0.66 / 0.67
        assert q.quote("T2") == (0.40, 0.41)                         # no order book available: falls back to the market object
        assert q.quote("T9") is None
    finally:
        K.BASE = old
        srv.shutdown()


def test_near_core_hours_use_a_lighter_shrinkage_and_the_radar_mirrors_the_engine():
    from trader import radar
    from trader.live import build_stats
    # which hours count as 'near' (circular), and only when core hours are in use
    a = SignalStats(mode="by_hour", hour_k=100, core_hours=(22, 23), core_k=10, near_k=50, near_width=2)
    assert a.near == frozenset({20, 21, 0, 1}) and a.k_for(22) == 10 and a.k_for(0) == 50 and a.k_for(21) == 50 and a.k_for(2) == 100 and a.k_for(19) == 100
    assert SignalStats(mode="by_hour", hour_k=100, near_k=50).near == frozenset() and SignalStats(mode="by_hour", hour_k=100, core_hours=(22,), core_k=10).near == frozenset()
    # the same hot evidence in a near hour (21), a far hour (16) and a core hour (22): near is trusted more than far, core most
    n = 24
    rows = []
    for hr in (16, 21, 22):
        for i in range(n):
            rows.append(dict(leg="DOG", hour=hr, close_ts=1_790_000_000 + i * 3600, open_ts=1_790_000_000 + i * 3600 - 900, win=int(i % 2 == 0), cost=0.2))
    for i in range(300):
        rows.append(dict(leg="DOG", hour=11, close_ts=1_790_000_000 + i * 600, open_ts=1_790_000_000 + i * 600 - 900, win=int(i % 5 == 0), cost=0.2))           # break-even pool elsewhere
    sig = pd.DataFrame(rows)
    now = 1_790_000_000 + 400 * 3600
    plain = SignalStats.from_signals(sig, mode="by_hour", hour_k=100, prior_n=20, basis="edge", core_hours=(22,), core_k=10)
    near = SignalStats.from_signals(sig, mode="by_hour", hour_k=100, prior_n=20, basis="edge", core_hours=(22,), core_k=10, near_k=40, near_width=2)
    p16, p21, p22 = (near.p("DOG", h, 0.2, now)[0] for h in (16, 21, 22))
    assert abs(near.p("DOG", 16, 0.2, now)[0] - plain.p("DOG", 16, 0.2, now)[0]) < 1e-12 and abs(p22 - plain.p("DOG", 22, 0.2, now)[0]) < 1e-12           # far and core hours are untouched
    assert near.p("DOG", 21, 0.2, now)[0] > plain.p("DOG", 21, 0.2, now)[0] + 0.02 and p22 > p21 > p16                                                    # the near hour reacts sooner than the far one
    # LiveConfig / build_stats pass the setting through (backtest does the same: BacktestConfig has the same two fields)
    cfg = LiveConfig(p_model="by_hour", hour_k=100.0, core_hours=(22, 23), core_k=10.0, near_k=50.0, near_width=2)
    assert build_stats(_parity_rows([0, 1]).assign(vol=np.nan), cfg).near == frozenset({20, 21, 0, 1}) and (BacktestConfig.__dataclass_fields__["near_k"].default == 0.0)
    # radar: with the engine's own shrinkage settings its est_edge is exactly what the engine's estimator gives (a radar 'TRADE' = the gate is open for a typical signal)
    rng = np.random.default_rng(3)
    recs = []
    for d in range(60):
        for hr in range(24):
            for q in range(4):
                o = int(datetime(2026, 8, 1, tzinfo=timezone.utc).timestamp()) + d * 86400 + hr * 3600 + q * 900
                hot = hr == 20 and d >= 40
                recs.append(dict(ticker=f"T{o}", open_ts=o, minute=12, bid=0.19, ask=0.20, yes=int(rng.random() < (0.55 if hot else 0.18))))
    rws = pd.DataFrame(recs)
    t_now = float(rws.open_ts.max()) + 3600
    rc = radar.RadarConfig(hour_k=100.0, halflife_days=14.0, core_hours=(22, 23), core_k=10.0, near_k=50.0, near_width=2)
    df = radar.hour_radar(rws, t_now, rc, (22, 23))
    from trader.data import extract_signals
    from trader.fees import fee_per_contract
    sg = extract_signals(rws, StrategyParams(hours=tuple(range(24)), minutes=rc.minutes, cost_min=rc.cost_min, cost_max=rc.cost_max, legs=("DOG",)))
    eng = SignalStats.from_signals(sg[sg.close_ts <= t_now], mode="by_hour", hour_k=100.0, prior_n=rc.prior_n, halflife_days=14.0, basis="edge", core_hours=(22, 23), core_k=10.0, near_k=50.0, near_width=2)
    typ = (rc.cost_min + rc.cost_max) / 2
    px = min(typ + rc.slippage, 0.99)
    for hr in (20, 5, 22):
        want = eng.p("DOG", hr, typ, t_now)[0] - (px + fee_per_contract(rc.fee_model, px))
        assert abs(float(df[df.hour == hr].iloc[0].est_edge) - want) < 1e-9, hr
    # and the near hour 20 (hot since day 40) is found sooner with the near-core shrinkage than with the plain scout shrinkage
    plain_df = radar.hour_radar(rws, t_now, radar.RadarConfig(hour_k=100.0, halflife_days=14.0, core_hours=(22, 23), core_k=10.0), (22, 23))
    assert float(df[df.hour == 20].iloc[0].est_edge) > float(plain_df[plain_df.hour == 20].iloc[0].est_edge) + 0.01



# ── the live engine replayed on history == the backtest ───────────────────────────
def _history(days, p_win, seed, vol=True):
    """A synthetic Kalshi history: every 15-minute window of every day, YES bid/ask .14/.15 at minutes 12 and 13, some windows with a wide quote at minute 12 (the minute-13 fallback decides) or no minute-12 row at all."""
    rng = np.random.default_rng(seed)
    base = int(datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp())
    recs = []
    for d in range(days):
        for w in range(96):
            o = base + d * 86400 + w * 900
            y = int(rng.random() < p_win)
            kind = rng.random()
            v = float(rng.choice([300.0, 4_000.0, 90_000.0])) if vol else np.nan
            if kind > 0.12:
                recs.append(dict(ticker=f"KXBTC15M-{o}", open_ts=o, minute=12, bid=0.14, ask=0.15, yes=y, vol=v))
            elif kind > 0.06:
                recs.append(dict(ticker=f"KXBTC15M-{o}", open_ts=o, minute=12, bid=0.45, ask=0.46, yes=y, vol=v))      # not a signal at 12 ...
            recs.append(dict(ticker=f"KXBTC15M-{o}", open_ts=o, minute=13, bid=0.14, ask=0.15, yes=y, vol=v))          # ... and the fallback minute
    return pd.DataFrame(recs)


def test_replay_of_the_live_engine_equals_the_backtest_on_synthetic_history():
    """The strongest form of 'live == backtest': drive the real LiveEngine through every window of a past period on a simulated clock and diff its trades with run_backtest's - including the daily-loss
    pause across midnight, the drawdown halt, the throttle, Auto sizing with a liquidity cap, a missing volume reading and the minute-13 fallback."""
    from trader.adaptive import AutoConfig
    from trader.replay import verify
    rows = _history(36, 0.30, 11)
    start, end = date(2026, 5, 21), date(2026, 6, 4)
    all_hours = tuple(range(24))
    base = dict(start=start, end=end, params=StrategyParams(hours=all_hours, cost_min=0.05, cost_max=0.30), capital=5_000.0)
    scenarios = {
        "history, by_hour": dict(p_model="by_hour", hour_k=40.0, lookback_days=30),
        "history, decaying memory + throttle off": dict(halflife_days=14.0, dd_throttle=False, min_edge=0.02),
        "fixed p, aggressive, daily-loss pause + drawdown halt": dict(prob_mode=PROB_MODES[1], fixed_p=0.40, kelly_mult=1.0, max_frac=0.08, daily_loss_limit=0.10, max_drawdown_stop=0.40),
        "auto, thin volume binds": dict(p_model="by_hour", hour_k=40.0, auto=AutoConfig(ladder=((0.0, 0.5, 0.03), (20_000.0, 0.3, 0.02)), participation=0.10)),
    }
    seen_pause = False
    for name, kw in scenarios.items():
        cmp, bt, rp = verify(rows, BacktestConfig(**{**base, **kw}))
        assert cmp.identical and len(bt.trades) >= 10, (name, cmp.text(), len(bt.trades))
        seen_pause |= bool(bt.metrics.get("windows_paused", 0))
    assert seen_pause                                                                      # the pause rule was really exercised
    # volume readings missing altogether: both sides fall back the same way (the default assumed volume)
    nov = _history(36, 0.30, 11, vol=False)
    cmp, bt, _rp = verify(nov, BacktestConfig(**{**base, "p_model": "by_hour", "hour_k": 40.0, "auto": AutoConfig(participation=0.10)}))
    assert cmp.identical and len(bt.trades) >= 10 and bt.trades.contracts.max() <= int(0.10 * 2000.0) + 1, (cmp.text(), bt.trades.contracts.max())


def test_replay_would_catch_a_backtest_that_forgot_the_settlement_delay():
    """Sensitivity check of the harness itself: a P&L that settles a few seconds after midnight belongs to the NEW day in the live engine; if the backtest booked it on the old day the two would disagree."""
    import trader.backtest as bt_mod
    from trader.replay import verify
    rows = _history(40, 0.18, 4)
    cfg = BacktestConfig(start=date(2026, 5, 11), end=date(2026, 6, 9), params=StrategyParams(hours=tuple(range(24)), cost_min=0.05, cost_max=0.30), capital=5_000.0,
                         prob_mode=PROB_MODES[1], fixed_p=0.40, kelly_mult=1.0, max_frac=0.04, daily_loss_limit=0.10, max_drawdown_stop=0.0)
    assert verify(rows, cfg)[0].identical
    keep = bt_mod.SETTLE_DELAY
    try:
        bt_mod.SETTLE_DELAY = 0.0
        broken = verify(rows, cfg)[0]
    finally:
        bt_mod.SETTLE_DELAY = keep
    assert not broken.identical, "the harness is blind to the day-start rule"


def test_replay_of_the_live_engine_equals_the_backtest_on_real_history():
    """The same proof on the real Kalshi candles: default R2+ preset and Auto, 6 weeks (skipped when the history is not on this machine)."""
    from trader.adaptive import AutoConfig
    from trader.data import DATA, load_rows
    from trader.replay import verify
    if not (DATA / "markets.csv").exists() or not (DATA / "candles.csv").exists():
        print("     (no local history - skipped)")
        return
    rows = load_rows()
    if rows.open_ts.max() < int(datetime(2026, 9, 30, tzinfo=timezone.utc).timestamp()):
        print("     (history too short - skipped)")
        return
    p = StrategyParams(hours=tuple(range(24)), minutes=(12, 13), cost_min=0.15, cost_max=0.35)
    r2 = dict(params=p, p_model="by_hour", hour_k=100.0, lookback_days=45, halflife_days=14.0, core_hours=(22, 23), core_k=10.0, near_k=50.0, near_width=2)
    for name, kw in {"R2+": r2, "AUTO": {**r2, "halflife_days": 0.0, "auto": AutoConfig()}}.items():
        cmp, bt, rp = verify(rows, BacktestConfig(start=date(2026, 8, 20), end=date(2026, 9, 30), capital=1_000.0, **kw))
        assert cmp.identical and len(bt.trades) >= 5, (name, cmp.text(), len(bt.trades))


def test_live_and_backtest_configs_share_one_set_of_defaults_and_map_both_ways():
    import dataclasses
    from trader.replay import backtest_config_from, live_config_from
    b0, l0 = BacktestConfig(start=date(2026, 1, 1), end=date(2026, 1, 2)), LiveConfig()
    renamed = {"slippage": "paper_slippage"}
    skip = {"start", "end", "start_ts", "end_ts", "prob_mode", "fixed_p", "compounding"}                 # mapped explicitly below (capital is one shared default: $1,000)
    for f in dataclasses.fields(b0):
        if f.name in skip:
            continue
        assert hasattr(l0, renamed.get(f.name, f.name)), f"LiveConfig has no counterpart of BacktestConfig.{f.name}"
        assert getattr(b0, f.name) == getattr(l0, renamed.get(f.name, f.name)), (f.name, getattr(b0, f.name), getattr(l0, renamed.get(f.name, f.name)))        # one default for the shared rule
    # round trip through the mapping
    b = BacktestConfig(start=date(2026, 3, 1), end=date(2026, 4, 1), capital=777.0, kelly_mult=0.5, slippage=0.03, p_model="by_hour", hour_k=100.0, core_hours=(22, 23), core_k=10.0, near_k=50.0,
                       lookback_days=30, daily_loss_limit=0.07)
    lc = live_config_from(b)
    assert lc.capital == 777.0 and lc.paper_slippage == 0.03 and lc.kelly_mult == 0.5 and lc.core_hours == (22, 23) and lc.daily_loss_limit == 0.07 and lc.prob_source == "history" and lc.mode == "paper"
    assert backtest_config_from(lc, b.start, b.end) == b
    fixed = live_config_from(BacktestConfig(start=b.start, end=b.end, prob_mode=PROB_MODES[1], fixed_p=0.33))
    assert fixed.prob_source == "manual" and fixed.manual_p == 0.33
    for bad in (dict(compounding=False), dict(prob_mode=PROB_MODES[2])):
        try:
            live_config_from(BacktestConfig(start=b.start, end=b.end, **bad))
            assert False, bad
        except ValueError:
            pass


def test_live_risk_limits_set_to_zero_are_off_like_in_the_backtest():
    """A limit of 0 means 'off' in the backtest; the live engine must not read it as 'halt immediately' (equity <= peak is always true at the start)."""
    cfg = LiveConfig(prob_source="manual", manual_p=0.40, paper_slippage=0.02, daily_loss_limit=0.0, max_drawdown_stop=0.0, capital=1000.0)
    eng, clock, q = _engine(cfg=cfg)
    clock.t = datetime(2026, 10, 9, 22, 12, 2, tzinfo=timezone.utc).timestamp()
    eng.tick()
    assert not eng.halted and len(eng._positions) == 1, (eng.halted, eng.halt_reason)


def test_resolve_volume_is_one_rule_for_the_live_engine_and_the_backtest():
    from trader.adaptive import VOLUME_FALLBACK, resolve_volume
    seen: list = []
    assert resolve_volume(None, seen) == (VOLUME_FALLBACK, False) and resolve_volume(float("nan"), seen) == (VOLUME_FALLBACK, False) and seen == []
    assert resolve_volume(500.0, seen) == (500.0, True) and resolve_volume(0.0, seen) == (0.0, True)           # a reading of 0 is a reading (nothing traded)
    assert resolve_volume(None, seen) == (250.0, False)                                                      # median of what was seen
    for v in range(1, 40):
        resolve_volume(float(v), seen)
    assert len(seen) == 20 and resolve_volume(None, seen)[0] == float(np.median(seen))


def test_compare_trades_names_every_difference():
    from trader.replay import compare_trades
    rows = _history(8, 0.3, 1)
    cfg = BacktestConfig(start=date(2026, 5, 3), end=date(2026, 5, 8), params=StrategyParams(hours=tuple(range(24)), cost_min=0.05, cost_max=0.30), prob_mode=PROB_MODES[1], fixed_p=0.40)
    t = run_backtest(rows, cfg).trades
    assert len(t) >= 5 and compare_trades(t, t.copy()).identical
    u = t.iloc[1:].copy().reset_index(drop=True)                                                              # one trade missing
    u.loc[0, "contracts"] += 1
    u.loc[1, "pnl"] += 0.5
    c = compare_trades(t, u)
    assert not c.identical and c.only_a == [t.market.iloc[0]] and not c.only_b
    assert {(d[0], d[1]) for d in c.diffs} >= {(u.market.iloc[0], "contracts"), (u.market.iloc[1], "pnl")}
    assert "DIFFERENT" in c.text() and "IDENTICAL" in compare_trades(t, t).text()



# ── session records and the live-vs-backtest reconcile ────────────────────────────
def test_session_records_and_settings_round_trip_through_the_ledger():
    from trader.adaptive import AutoConfig
    from trader.ledger import Ledger
    from trader.live import config_from_dict, config_to_dict
    cfg = LiveConfig(capital=777.0, params=StrategyParams(hours=(22, 23), minutes=(12, 13), cost_min=0.15, cost_max=0.35, legs=("DOG", "FAV")), core_hours=(22, 23), core_k=10.0, near_k=50.0, p_model="by_hour",
                     auto=AutoConfig(ladder=((0.0, 0.5, 0.03), (25_000.0, 0.35, 0.025)), half_lives=(7.0, 14.0, 30.0)), min_history=12)
    back = config_from_dict(json.loads(json.dumps(config_to_dict(cfg))))                      # through real JSON: tuples come back as lists and must be rebuilt
    assert back == cfg and isinstance(back.params.hours, tuple) and isinstance(back.auto.ladder[0], tuple)
    assert config_from_dict({**config_to_dict(LiveConfig()), "from_a_newer_version": 1}) == LiveConfig()          # unknown keys are ignored
    d = Path(tempfile.mkdtemp())
    eng, clock, led = _ledger_engine(d / "ledger.jsonl")
    w0 = clock.t
    for dt in (1, 12 * 60 + 2, 900 + 25, 900 + 600):                                              # first tick, the decision, the settlement, the next window
        clock.t = w0 + dt
        eng.quotes.res = "yes"
        eng.tick()
    eng.close("stopped by user")
    eng.close("again")                                                                            # closing twice writes one end
    ss = led.sessions()
    assert len(ss) == 1 and ss[0]["mode"] == "paper" and ss[0]["start_ts"] == w0 + 1 and ss[0]["end_ts"] == w0 + 900 + 600 and ss[0]["why"] == "stopped by user"
    assert config_from_dict(ss[0]["cfg"]) == eng.cfg and ss[0]["capital"] == eng.cfg.capital
    states = [e["state"] for e in led.read_events()[0] if e["ev"] == "session"]
    assert states == ["start", "alive", "end"], states                                             # one heartbeat per window, not per tick
    assert led.load().count() == 1 and led.load().bad_lines == 0                                  # session events are not trades and not 'unreadable lines'
    op = [e for e in led.read_events()[0] if e["ev"] == "open"][0]
    assert op["requested"] == op["contracts"] and led.load().settled[0]["requested"] == op["contracts"]
    # a halted session ends at the halt, and keeps its settings
    eng2, clock2, led2 = _ledger_engine(d / "ledger2.jsonl")
    clock2.t += 1
    eng2.tick()
    eng2.halt("test halt")
    clock2.t += 900 * 3
    eng2.tick()
    s2 = led2.sessions()[0]
    assert s2["end_ts"] == clock2.t - 900 * 3 and s2["why"].startswith("HALTED") and s2["last_ts"] == s2["end_ts"]


def _replayed_ledger(rows, cfg, broker=None, rows_live=None):
    """Replay the live engine on ``rows_live`` (default ``rows``) into a throw-away ledger, as a stand-in for a real session."""
    from trader.ledger import Ledger
    from trader.replay import replay_live
    led = Ledger(Path(tempfile.mkdtemp()) / "ledger.jsonl")
    replay_live(rows if rows_live is None else rows_live, cfg, ledger=led, broker=broker)
    return led


def test_reconcile_of_a_session_that_followed_the_backtest_exactly():
    from trader.reconcile import reconcile
    rows = _history(30, 0.30, 5)
    cfg = BacktestConfig(start=date(2026, 5, 11), end=date(2026, 5, 20), params=StrategyParams(hours=tuple(range(24)), cost_min=0.05, cost_max=0.30), capital=2_000.0, p_model="by_hour", hour_k=40.0)
    led = _replayed_ledger(rows, cfg)
    rec = reconcile(led, rows, mode="paper")
    s = rec.summary()
    n_bt = len(run_backtest(rows, cfg).trades)
    assert len(rec.sessions) == 1 and not rec.skipped and n_bt >= 20, (rec.skipped, n_bt)
    assert rec.sessions[0].windows == 10 * 96
    assert s["matched"] == n_bt and s["live_only"] == 0 and s["bt_only"] == 0 and s["agreement"] == 1.0, s
    assert abs(s["slippage_c"] - 2.0) < 1e-6 and abs(s["assumed_slippage_c"] - 2.0) < 1e-6 and s["fill_ratio"] == 1.0 and s["quote_diff_c"] == 0.0
    assert abs(s["pnl_live"] - s["pnl_bt"]) < 1e-6 and abs(s["ev_live_c"] - s["ev_bt_c"]) < 1e-9
    t = rec.table()
    assert (t.status == "matched").all() and list(t.window) == sorted(t.window) and "IDENTICAL" not in rec.text() and "agreement 100%" in rec.text()


def test_reconcile_measures_what_the_real_market_changed():
    """A session whose fills are worse than assumed, partly filled, and whose quotes sometimes differ from the candle closes: the report must say so, number by number."""
    from trader.reconcile import reconcile
    rows = _history(30, 0.30, 5)
    cfg = BacktestConfig(start=date(2026, 5, 11), end=date(2026, 5, 20), params=StrategyParams(hours=tuple(range(24)), cost_min=0.05, cost_max=0.30), capital=2_000.0, p_model="by_hour", hour_k=40.0)

    class WorseFills(PaperBroker):
        """Pays 1c more than assumed and gets 75% of the contracts."""
        def buy(self, ticker, side, contracts, max_cost, quoted_cost):
            n = max(1, int(contracts * 0.75))
            px = min(quoted_cost + self.slip + 0.01, 0.99)
            return Fill(qty=n, price=px, fee=fee_total(self.fee_model, n, px), order_id="paper", state="PAPER")

    live_rows = rows.copy()
    tickers = sorted(live_rows.ticker.unique())
    wide = set(tickers[1000:1030])                                                              # live saw a wide book at the decision minute in 30 windows ...
    live_rows.loc[live_rows.ticker.isin(wide) & (live_rows.minute == 12), ["bid", "ask"]] = [0.45, 0.46]
    live_rows.loc[live_rows.ticker.isin(wide) & (live_rows.minute == 13), ["bid", "ask"]] = [0.45, 0.46]
    led = _replayed_ledger(rows, cfg, broker=WorseFills("Kalshi", cfg.slippage), rows_live=live_rows)
    rec = reconcile(led, rows, mode="paper")                                                   # ... while the candles the backtest uses say there was a signal
    s = rec.summary()
    assert s["bt_only"] >= 3 and s["matched"] >= 20 and s["agreement"] < 1.0, s
    assert abs(s["slippage_c"] - 3.0) < 1e-6 and abs(s["assumed_slippage_c"] - 2.0) < 1e-6, s
    assert 0.6 < s["fill_ratio"] < 0.9, s["fill_ratio"]
    assert s["pnl_live"] != s["pnl_bt"] and set(rec.table().status) >= {"matched", "backtest only"}
    assert "backtest only" in rec.text() and "3.0c above the quote" in rec.text()


def test_reconcile_skips_what_it_cannot_compare_and_waits_for_data():
    from trader.ledger import Ledger
    from trader.reconcile import reconcile
    rows = _history(12, 0.30, 5)
    cfg = BacktestConfig(start=date(2026, 5, 3), end=date(2026, 5, 10), params=StrategyParams(hours=tuple(range(24)), cost_min=0.05, cost_max=0.30), capital=2_000.0)
    led = _replayed_ledger(rows, cfg)
    # the candle data stops three days before the session did: compare what exists and say so
    short = rows[rows.open_ts < int(datetime(2026, 5, 8, tzinfo=timezone.utc).timestamp())]
    rec = reconcile(led, short, mode="paper")
    assert len(rec.sessions) == 1 and rec.sessions[0].notes and "update the data" in rec.sessions[0].notes[0] and rec.sessions[0].windows == 5 * 96
    # a session with trades but no recorded settings (older version) is listed as skipped, never silently dropped
    old = Ledger(Path(tempfile.mkdtemp()) / "ledger.jsonl")
    old.record_open(mode="paper", session="OLD", market="KXBTC15M-OLD", window=1_780_000_000, leg="DOG", side="YES", quoted=0.15, entry=0.17, contracts=10, stake=1.8, fee=0.1, p_used=0.3, f_full=0.1, f_used=0.02,
                    bankroll_before=100.0, order_id="paper", entry_ts=1_780_000_720)
    rec = reconcile(old, rows, mode="paper")
    assert not rec.sessions and rec.skipped and "no recorded settings" in rec.skipped[0] and "Nothing to compare" in rec.text()
    assert not reconcile(Ledger(Path(tempfile.mkdtemp()) / "none.jsonl"), rows).sessions



# ── exchange shards: the 2026-10-12 22:42 incident (cash on shard 0, BTC 15m orders funded from shard 2) ──
def test_orders_are_funded_from_the_exchange_shard_of_the_market_not_from_the_total_balance():
    from trader.broker import InsufficientFunds
    srv, url = _mock_server()
    try:
        MockKalshi.shards, MockKalshi.series_index = {0: 1004.33, 2: 0.0}, 2                       # all the cash sits on the default shard; BTC 15m trades on shard 2 (Crypto & Commodities)
        q = kalshi.KalshiQuotes(base=url)
        assert q.exchange_index() == 2 and q.shard_names()[2] == "Crypto & Commodities"
        c = _client(url)
        tot, mine = c.balance(), c.balance(2)
        assert tot["buying_power"] == 1004.33 and tot["breakdown"] == {0: 1004.33, 2: 0.0} and mine["buying_power"] == 0.0 and mine["exchange_index"] == 2
        naive = KalshiBroker(c, fill_wait_s=1, exchange_index=None)                                 # what the first version did: it saw $1,004.33 and sent the order ...
        assert naive.buying_power() == 1004.33
        try:
            naive.buy(MockKalshi.ticker, "YES", 50, 0.35, 0.32)                                     # ... and Kalshi refused it for insufficient balance
            assert False
        except InsufficientFunds as e:
            assert isinstance(e, NotPlaced)
        br = KalshiBroker(c, fill_wait_s=1, exchange_index=2)
        assert br.buying_power() == 0.0
        try:
            br.buy(MockKalshi.ticker, "YES", 50, 0.35, 0.32)
            assert False
        except InsufficientFunds as e:
            msg = str(e)
            assert "shard 2" in msg and "$0.00" in msg and "$1,004.33" in msg and "Exchange Indexes" in msg and "shard 0: $1,004.33" in msg, msg
        assert MockKalshi.created[-1]["exchange_index"] == 2 and MockKalshi.created[-1]["side"] == "bid" and MockKalshi.created[-1]["price"] == "0.3500"      # the order is routed straight to the market's shard
        try:                                                                                          # the order-path test explains itself too
            br.order_path_test(MockKalshi.ticker)
            assert False
        except InsufficientFunds as e:
            assert "shard 2" in str(e)
        MockKalshi.shards[2] = 50.0                                                                  # cash moved to the shard: the same order now trades
        assert br.buying_power() == 50.0
        f = br.buy(MockKalshi.ticker, "YES", 40, 0.20, 0.17)
        assert f.qty == 40 and f.order_id
        # a NO (an ask on the YES book) needs 1 - price per contract of collateral: 100 contracts at NO 0.30 = $30 fits in $50, 200 do not
        assert br.buy(MockKalshi.ticker, "NO", 100, 0.30, 0.28).qty == 100
        try:
            br.buy(MockKalshi.ticker, "NO", 200, 0.30, 0.28)
            assert False
        except InsufficientFunds:
            pass
        # by default the broker looks the shard up itself, so a caller cannot forget it
        MockKalshi.shards = {0: 4.33, 2: 80.0}
        auto = KalshiBroker(c, fill_wait_s=1)
        assert auto.exchange_index == 2 and auto.buying_power() == 80.0 and auto.buy(MockKalshi.ticker, "YES", 5, 0.2, 0.17).qty == 5 and MockKalshi.created[-1]["exchange_index"] == 2
        # a market whose shard Kalshi does not report: no exchange_index is sent (Kalshi auto-routes by ticker) and the total balance is used
        MockKalshi.shards, MockKalshi.series_index = {0: 950.5}, None
        unknown = KalshiBroker(c, fill_wait_s=1)
        assert unknown.exchange_index is None and unknown.buying_power() == 950.5 and unknown.buy(MockKalshi.ticker, "YES", 5, 0.2, 0.17).qty == 5 and "exchange_index" not in MockKalshi.created[-1]
    finally:
        srv.shutdown()


def test_live_engine_survives_an_insufficient_balance_rejection_alerts_and_trades_again_once_funded():
    """The incident, reproduced: the engine saw $1,004 (total), sized a normal order, Kalshi said insufficient_balance.  It must stay alive, say why (alert + ledger), and trade the next signal once the cash is on the right shard."""
    srv, url = _mock_server()
    try:
        MockKalshi.shards, MockKalshi.series_index = {0: 1004.33, 2: 0.0}, 2
        d = Path(tempfile.mkdtemp())
        from trader.ledger import Ledger
        eng, clock, events = _engine(cfg=LiveConfig(capital=1000.0), broker=KalshiBroker(_client(url), fill_wait_s=1, exchange_index=None))            # the old, shard-blind broker
        eng.ledger = Ledger(d / "ledger.jsonl")
        w0 = clock.t
        clock.t = w0 + 12 * 60 + 2
        eng.tick()
        alerts = [e for k, e in events if k == "alert"]
        assert not eng.halted and not eng._positions and len(alerts) == 1 and "shard 2" in alerts[0]["msg"] and "insufficient balance" in alerts[0]["msg"], (eng.halt_reason, alerts)
        assert any("ORDER NOT PLACED" in e["msg"] for k, e in events if k == "log")
        sess_alerts = [e for e in eng.ledger.read_events()[0] if e["ev"] == "session" and e["state"] == "alert"]
        assert len(sess_alerts) == 1 and "shard 2" in sess_alerts[0]["why"]                                                  # the ledger remembers why a window had no trade
        MockKalshi.shards[2] = 300.0                                                                                          # the user moves cash to shard 2
        clock.t = w0 + 900 + 12 * 60 + 2
        eng.tick()
        assert len(eng._positions) == 1 and not eng.halted
    finally:
        srv.shutdown()


def test_live_engine_says_so_when_the_markets_shard_has_no_cash_at_all():
    srv, url = _mock_server()
    try:
        MockKalshi.shards, MockKalshi.series_index = {0: 1004.33, 2: 0.0}, 2
        br = KalshiBroker(_client(url), fill_wait_s=1, exchange_index=2)                                                     # the shard-aware broker reads $0 and sizes nothing ...
        eng, clock, events = _engine(cfg=LiveConfig(capital=1000.0), broker=br)
        clock.t += 12 * 60 + 2
        eng.tick()
        alerts = [e for k, e in events if k == "alert"]
        assert not eng.halted and not eng._positions and len(alerts) == 1 and "no order can be placed" in alerts[0]["msg"] and not MockKalshi.created          # ... and says so, without sending anything
    finally:
        srv.shutdown()



# ── auto-fund: the API does what the Kalshi app does before an order ──────────────
def test_broker_fund_moves_cash_between_own_shards_exactly_and_safely():
    from trader.broker import BrokerError
    srv, url = _mock_server()
    try:
        MockKalshi.shards, MockKalshi.series_index = {0: 1000.0, 1: 3.0, 2: 4.33}, 2
        c = _client(url)
        br = KalshiBroker(c, fill_wait_s=1, exchange_index=2)
        c.transfer(10.0, 0, 2, "fixed-id")
        t = MockKalshi.transfers[-1]
        assert t["amount"] == 100_000 and t["source_exchange_shard"] == 0 and t["destination_exchange_shard"] == 2 and t["source"] == t["destination"] == "event_contract"     # centicents
        c.transfer(10.0, 0, 2, "fixed-id")                                                           # the same id again: Kalshi does not execute it twice
        assert len(MockKalshi.transfers) == 1 and MockKalshi.shards[0] == 990.0
        r = br.fund(100.0)                                                                            # exactly what is needed, from the biggest balance
        assert r["moved"] == 100.0 and r["arrived"] and r["moves"] == [(0, 100.0)] and abs(MockKalshi.shards[2] - 114.33) < 1e-9 and abs(MockKalshi.shards[0] - 890.0) < 1e-9
        r = br.fund(5000.0)                                                                           # more than exists elsewhere: everything elsewhere, biggest first, nothing from the destination
        assert r["moves"] == [(0, 890.0), (1, 3.0)] and r["moved"] == 893.0 and MockKalshi.shards[0] == 0.0 and MockKalshi.shards[1] == 0.0 and abs(MockKalshi.shards[2] - 1007.33) < 1e-9
        r = br.fund(50.0)                                                                             # nothing left elsewhere: no call, no error
        assert r["moved"] == 0.0 and not r["arrived"] and len(MockKalshi.transfers) == 4          # (1 explicit + 1 + 2 from the earlier funds): nothing new
        MockKalshi.shards, MockKalshi._pending = {0: 100.0, 2: 0.0}, []                              # Kalshi completes transfers asynchronously: wait for the cash to show up ...
        MockKalshi.transfer_lag = 3
        r = KalshiBroker(c, fill_wait_s=1, exchange_index=2).fund(40.0, wait_s=6.0)
        assert r["arrived"] and r["moved"] == 40.0 and MockKalshi.shards[2] == 40.0
        MockKalshi.shards, MockKalshi._pending, MockKalshi.transfer_lag = {0: 100.0, 2: 0.0}, [], 500                    # ... but not forever, and it says it has not arrived
        r = KalshiBroker(c, fill_wait_s=1, exchange_index=2).fund(40.0, wait_s=1.0)
        assert r["moved"] == 40.0 and not r["arrived"]
        MockKalshi.transfer_lag, MockKalshi.transfer_mode = 0, "reject400"
        try:
            KalshiBroker(c, fill_wait_s=1, exchange_index=2).fund(10.0)
            assert False
        except BrokerError as e:
            assert "transfer refused" in str(e) and "shard 0" in str(e)
        MockKalshi.transfer_mode = "ok"
        try:
            KalshiBroker(c, fill_wait_s=1, exchange_index=None).fund(10.0)                           # no shard known: nothing to fund
            assert False
        except BrokerError:
            pass
    finally:
        srv.shutdown()


def _fund_engine(url, **cfg):
    br = KalshiBroker(_client(url), fill_wait_s=1, exchange_index=2)
    eng, clock, events = _engine(cfg=LiveConfig(capital=1000.0, **cfg), broker=br)
    return eng, clock, events


def test_auto_fund_tops_the_trading_shard_up_before_the_window_and_the_order_then_trades():
    srv, url = _mock_server()
    try:
        MockKalshi.shards, MockKalshi.series_index = {0: 1000.0, 2: 4.33}, 2                          # exactly the user's account on 2026-10-12
        eng, clock, events = _fund_engine(url, auto_fund=True)
        w0 = clock.t
        clock.t = w0 + 5
        eng.tick()                                                                                    # session start: funded outside the decision second
        assert len(MockKalshi.transfers) == 1 and abs(MockKalshi.shards[2] - 1000.0) < 1e-6 and abs(MockKalshi.shards[0] - 4.33) < 1e-6
        assert any("auto-fund (session start)" in e["msg"] for k, e in events if k == "log")
        clock.t = w0 + 12 * 60 + 2
        eng.tick()
        assert len(eng._positions) == 1 and not eng.halted and not [e for k, e in events if k == "alert"]
        assert MockKalshi.created[-1]["exchange_index"] == 2 and len(MockKalshi.transfers) == 1             # the order itself did not wait for a transfer
        clock.t = w0 + 900 + 40
        eng.tick()
        assert len(MockKalshi.transfers) == 1                                                         # already funded: no second transfer
    finally:
        srv.shutdown()


def test_auto_fund_off_never_moves_cash_and_fund_max_caps_what_it_moves():
    srv, url = _mock_server()
    try:
        MockKalshi.shards, MockKalshi.series_index = {0: 1000.0, 2: 4.33}, 2
        eng, clock, events = _fund_engine(url, auto_fund=False)
        clock.t += 5; eng.tick(); clock.t += 12 * 60; eng.tick()
        assert MockKalshi.transfers == [] and abs(MockKalshi.shards[2] - 4.33) < 1e-9                      # nothing moved; the 4.33 bankroll sizes no bet
        MockKalshi.shards = {0: 1000.0, 2: 4.33}
        eng, clock, events = _fund_engine(url, auto_fund=True, fund_max=200.0)
        clock.t += 5; eng.tick()
        assert abs(MockKalshi.shards[2] - 200.0) < 1e-6 and abs(MockKalshi.shards[0] - 804.33) < 1e-6      # keeps at most $200 on the trading shard
        assert LiveConfig().auto_fund is False                                                          # the engine default is OFF; the GUI decides
    finally:
        srv.shutdown()


def test_auto_fund_inside_the_decision_second_is_only_the_fallback():
    """Session started right at the decision instant and the shard is empty: fund, re-read the cash, size, order."""
    srv, url = _mock_server()
    try:
        MockKalshi.shards, MockKalshi.series_index = {0: 1000.0, 2: 0.0}, 2
        eng, clock, events = _fund_engine(url, auto_fund=True)
        clock.t += 12 * 60 + 2
        eng.tick()
        assert len(MockKalshi.transfers) == 1 and len(eng._positions) == 1 and not eng.halted
        assert any("auto-fund (no cash at the decision instant)" in e["msg"] for k, e in events if k == "log")
    finally:
        srv.shutdown()


def test_auto_fund_problems_are_reported_never_fatal():
    srv, url = _mock_server()
    try:
        MockKalshi.shards, MockKalshi.series_index = {0: 1000.0, 2: 0.0}, 2
        MockKalshi.transfer_mode = "reject400"
        eng, clock, events = _fund_engine(url, auto_fund=True)
        clock.t += 5
        eng.tick()
        alerts = [e for k, e in events if k == "alert"]
        assert not eng.halted and len(alerts) == 1 and "Auto-fund failed" in alerts[0]["title"] and "transfer refused" in alerts[0]["msg"]
        MockKalshi.transfer_mode, MockKalshi.shards = "ok", {0: 0.0, 2: 0.0}                              # no cash anywhere: the loud 'no order can be placed' alert, no halt
        eng, clock, events = _fund_engine(url, auto_fund=True)
        clock.t += 12 * 60 + 2
        eng.tick()
        assert not eng.halted and any("no order can be placed" in e["msg"] for k, e in events if k == "alert") and not MockKalshi.created
    finally:
        srv.shutdown()


def test_order_rejected_for_cash_is_retried_once_after_funding_and_never_loops():
    from trader.broker import InsufficientFunds

    class Stub:
        mode, venue = "live", "kalshi"
        def __init__(self, funded_after):
            self.fund_calls, self.buys, self.funded, self.funded_after = [], 0, False, funded_after
        def buying_power(self): return 1000.0
        def fund(self, amount, wait_s=15.0):
            self.fund_calls.append(amount); self.funded = True
            return dict(moved=amount, arrived=True, balance=1000.0, moves=[(0, amount)])
        def buy(self, ticker, side, n, mx, q):
            self.buys += 1
            if not self.funded and self.funded_after:
                raise InsufficientFunds("cash is on another shard")
            if not self.funded_after:
                raise InsufficientFunds("still no cash")
            return Fill(n, 0.17, 0.2, "oid-9", "FILLED")
        def cancel_all(self): pass

    for funded_after, want_pos in ((True, 1), (False, 0)):
        stub = Stub(funded_after)
        eng, clock, events = _engine(cfg=LiveConfig(capital=1000.0, auto_fund=True), broker=stub)
        clock.t += 12 * 60 + 2
        eng.tick()
        assert len(eng._positions) == want_pos and not eng.halted and stub.buys == 2 and len(stub.fund_calls) == 1, (funded_after, stub.buys, stub.fund_calls)       # exactly one retry
        assert (not want_pos) == bool([e for k, e in events if k == "alert"])
    stub = Stub(True)
    eng, clock, events = _engine(cfg=LiveConfig(capital=1000.0, auto_fund=False), broker=stub)            # auto-fund off: no transfer, the rejection is just reported
    clock.t += 12 * 60 + 2
    eng.tick()
    assert not eng._positions and stub.buys == 1 and not stub.fund_calls



# ── HTTP 429: the app must not cause it, and must survive it ─────────────────────
def test_rate_limiter_spaces_requests_and_never_blocks_for_long():
    from trader.kalshi import RateLimiter
    rl = RateLimiter(rate=20.0, burst=2)
    t0 = time.monotonic()
    for _ in range(6):
        rl.acquire(5.0)
    el = time.monotonic() - t0
    assert 0.15 <= el < 1.0, el                                                         # 2 free (burst) + 4 at 20/s = about 0.2 s: spaced, not blocked
    rl2 = RateLimiter(rate=0.5, burst=1)
    rl2.acquire(1.0)
    t0 = time.monotonic()
    rl2.acquire(0.05)                                                                   # empty bucket, 2 s to the next token: gives up waiting after 50 ms and lets the request go
    assert time.monotonic() - t0 < 0.5


class _CountingQuotes(FakeQuotes):
    """Quote source whose market lookup / book reads can be switched off; counts every request."""
    def __init__(self, fail_s, clock):
        super().__init__()
        self.fail_s, self.clock, self.t0, self.calls = fail_s, clock, None, 0
    def window_ticker(self, open_ts):
        self.calls += 1
        return None if self.clock.t - self.t0 < self.fail_s else super().window_ticker(open_ts)


def test_engine_backs_off_when_a_quote_is_unavailable_instead_of_hammering_the_api():
    """A failed read at the decision instant used to be retried every ~5 ms for the whole 25 s grace period (hundreds of requests per second in this simulation) - the way to HTTP 429."""
    for fail_s, max_calls, trades in ((4.0, 14, 1), (25.0, 40, 0)):
        eng, clock, events = _engine(cfg=LiveConfig(capital=1000.0), quotes=None)
        q = _CountingQuotes(fail_s, clock)
        eng.quotes = q
        q.t0 = clock.t + 12 * 60
        t, end = clock.t + 12 * 60 - 20, clock.t + 12 * 60 + 40
        while t < end:                                                                      # the LiveWorker's loop: tick, then sleep as long as the engine says
            clock.t = t
            eng.tick()
            t += max(0.005, min(0.25, eng.sleep_hint(t)))
        assert q.calls <= max_calls and len(eng._positions) == trades, (fail_s, q.calls, len(eng._positions))
    # the book is read once per attempt: when it fails the fallback is the market quote, NOT a second book request
    class BookDown:
        def __init__(self): self.books = self.market = self.quote_calls = 0
        def window_ticker(self, o): return f"KXBTC15M-{o}"
        def book(self, t): self.books += 1; return None
        def market_quote(self, t): self.market += 1; return (0.14, 0.15)
        def quote(self, t): self.quote_calls += 1; return (0.14, 0.15)
        def result(self, t): return "no"
    bq = BookDown()
    eng, clock, events = _engine(cfg=LiveConfig(capital=1000.0), quotes=bq)
    clock.t += 12 * 60 + 1
    eng.tick()
    assert bq.books == 1 and bq.market == 1 and bq.quote_calls == 0 and len(eng._positions) == 1


def test_window_market_lookup_is_negative_cached_and_the_ticker_is_warmed_up_early():
    class Resp:
        status_code = 200
        def json(self): return {"markets": []}
    class Sess:
        def __init__(self): self.n = 0
        def get(self, *a, **k): self.n += 1; return Resp()
    s = Sess()
    q = kalshi.KalshiQuotes(session=s, base="http://127.0.0.1:9")
    assert q.window_ticker(1_780_000_000) is None and s.n == 1
    assert q.window_ticker(1_780_000_000) is None and s.n == 1                              # not asked again within a second
    # the engine looks the window's market up early (20-300 s into the window), so the decision second only reads the book
    seen = []
    class Q(FakeQuotes):
        def window_ticker(self, o): seen.append(o); return super().window_ticker(o)
    eng, clock, events = _engine(cfg=LiveConfig(capital=1000.0), quotes=Q())
    w0 = clock.t
    for dt in (5, 25, 26, 200, 400):
        clock.t = w0 + dt
        eng.tick()
    assert seen == [int(w0)]                                                                 # once, at +25 s (not at +5 s, not again later)


def test_a_failing_cash_read_never_halts_the_session():
    from trader.kalshi import ApiError

    class Flaky:
        mode = "live"
        def __init__(self, fails): self.fails, self.reads, self.bought = fails, 0, 0
        def buying_power(self):
            self.reads += 1
            if self.fails > 0:
                self.fails -= 1
                raise ApiError(429, "too many requests")
            return 500.0
        def buy(self, t, side, n, mx, q): self.bought += 1; return Fill(n, 0.17, 0.2, "o1", "FILLED")
        def cancel_all(self): pass
    # two 429s, then fine: re-read within the decision, trade as normal
    b = Flaky(2)
    eng, clock, events = _engine(cfg=LiveConfig(capital=1000.0), broker=b)
    clock.t += 12 * 60 + 2
    eng.tick()
    assert not eng.halted and b.reads == 3 and b.bought == 1 and len(eng._positions) == 1
    # a persistent failure with no earlier reading: the signal is tracked, no order, NO halt
    b = Flaky(99)
    eng, clock, events = _engine(cfg=LiveConfig(capital=1000.0), broker=b)
    clock.t += 12 * 60 + 2
    eng.tick()
    assert not eng.halted and b.bought == 0 and not eng._positions and any("could not be read" in e["msg"] for k, e in events if k == "log")
    assert len(eng._shadow) == 1                                                             # the signal still teaches the model
    # a persistent failure after an earlier good reading: the last known cash is used
    b = Flaky(0)
    eng, clock, events = _engine(cfg=LiveConfig(capital=1000.0), broker=b)
    eng._last_bp = (500.0, clock.t - 60)
    b.fails = 99
    clock.t += 12 * 60 + 2
    eng.tick()
    assert not eng.halted and b.bought == 1 and any("last known" in e["msg"] for k, e in events if k == "log")


def test_client_re_sends_reads_on_429_but_never_writes():
    from trader.kalshi import ApiError
    srv, url = _mock_server()
    try:
        c = _client(url)
        MockKalshi.throttle_gets = 2                                                          # two 429s (Retry-After: 0), then OK: the read succeeds on its third try
        assert c.balance()["buying_power"] == 950.5
        assert MockKalshi.paths.count("/trade-api/v2/portfolio/balance") == 3
        MockKalshi.paths.clear()
        MockKalshi.throttle_gets = 99                                                         # persistent: gives up after exactly 3 tries with a message that says what it is
        try:
            c.balance()
            assert False
        except ApiError as e:
            assert e.status == 429 and "rate limit" in e.message and MockKalshi.paths.count("/trade-api/v2/portfolio/balance") == 3
        MockKalshi.throttle_gets = 0
        MockKalshi.mode = "rate429"                                                           # an ORDER answered with 429 is not re-sent by the client (the caller skips the signal)
        MockKalshi.created.clear()
        br = KalshiBroker(c, fill_wait_s=1, exchange_index=None)
        try:
            br.buy(MockKalshi.ticker, "YES", 5, 0.2, 0.17)
            assert False
        except NotPlaced:
            pass
        assert len(MockKalshi.created) == 1
    finally:
        srv.shutdown()


def test_downloads_run_slowly_while_a_live_session_is_running():
    import trader.data as D
    seen = {}
    class P:
        stdout = iter([])
        def wait(self): return 0
    orig = D.subprocess.Popen
    D.subprocess.Popen = lambda *a, **k: (seen.update(env=k.get("env")), P())[1]
    try:
        import pandas as _pd
        old_read = D.pd.read_csv
        D.pd.read_csv = lambda *a, **k: _pd.DataFrame({"open_ts": [1_780_000_000]})
        try:
            D.update_data(lambda s: None, rps=1.0)
            assert seen["env"]["COLLECT_RPS"] == "1.0"
            seen.clear()
            D.update_data(lambda s: None)
            assert "COLLECT_RPS" not in seen["env"] or seen["env"].get("COLLECT_RPS") == os.environ.get("COLLECT_RPS")
        finally:
            D.pd.read_csv = old_read
    finally:
        D.subprocess.Popen = orig



# ── EV-regime filter presets ──────────────────────────────────────────────────────
def _hourly_history(days, seed, hot_hours=(0, 1, 2, 3, 4, 5), p_hot=0.55, p_cold=0.10):
    """Every window of every day with a 15c underdog; the hours in ``hot_hours`` really are good (EV about +35c), the rest really are bad (about -9c)."""
    rng = np.random.default_rng(seed)
    base = int(datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp())
    recs = []
    for d in range(days):
        for w in range(96):
            o = base + d * 86400 + w * 900
            hr = (w * 900) // 3600
            y = int(rng.random() < (p_hot if hr in hot_hours else p_cold))
            for m in (12, 13):
                recs.append(dict(ticker=f"KXBTC15M-{o}", open_ts=o, minute=m, bid=0.14, ask=0.15, yes=y, vol=5_000.0))
    return pd.DataFrame(recs)


def test_trailing_ev_window_equals_the_radars_numbers_and_has_no_lookahead():
    from trader import radar
    from trader.data import extract_signals
    from trader.evfilter import TrailingEVStats
    rows = _hourly_history(45, 3)
    p = StrategyParams(hours=tuple(range(24)), minutes=(12, 13), cost_min=0.15, cost_max=0.35, legs=("DOG",))
    sig = extract_signals(rows, p)
    now = float(sig.close_ts.max()) - 5 * 86400
    st = TrailingEVStats.from_signals(sig[sig.close_ts <= now], ((14, 0.10),), 8, 0.02, "Kalshi")        # built from the PAST only
    full = TrailingEVStats.from_signals(sig, ((14, 0.10),), 8, 0.02, "Kalshi")                           # built with the future in it as well
    df = radar.hour_radar(rows, now, radar.RadarConfig(window_days=28), ())
    for h in range(24):
        r = df[df.hour == h].iloc[0]
        n28, ev28, _g = st.window(h, now, 28)
        n14, ev14, _g14 = st.window(h, now, 14)
        assert n28 == r.n and n14 == r.n14 and abs(ev28 - r.ev) < 1e-12 and abs(ev14 - r.ev14) < 1e-12, (h, n28, r.n, ev28, r.ev)       # 'Net EV 28 d / 14 d' of the radar
        fn28, fev28, _ = full.window(h, now, 28)
        assert fn28 == n28 and abs(fev28 - ev28) < 1e-12                                              # later signals never change what was known at ``now``


def test_ev_rules_pass_fail_min_signals_and_sizing_probability():
    from trader.evfilter import TrailingEVStats
    day = 86400
    t0 = 1_780_000_000 // 900 * 900
    def hist(hour_wins, hour=5, cost=0.15, step=day // 4):
        rows = []
        for i, w in enumerate(hour_wins):
            o = t0 + i * step
            rows.append(dict(leg="DOG", hour=hour, open_ts=o, close_ts=o + 900, win=w, cost=cost))
        return pd.DataFrame(rows)
    sig = hist([1] * 12 + [0] * 4)                                                                    # 16 signals over 4 days: 75% win at 15c = a big edge
    st = TrailingEVStats.from_signals(sig, ((14, 0.10),), 8, 0.02, "Kalshi")
    now = float(sig.close_ts.max()) + 60
    ok, gross, parts = st.check(5, now)
    n, ev, g = st.window(5, now, 14)
    assert ok and n == 16 and ev > 0.10 and abs(gross - g) < 1e-12 and abs(g - (0.75 - 0.15)) < 1e-9
    p, n_leg, nh = st.p("DOG", 5, 0.20, now)
    assert abs(p - (0.20 + gross)) < 1e-12 and n_leg == 16 and nh == 16                               # a passing hour is sized off its own trailing gross edge
    assert st.p("DOG", 6, 0.20, now)[0] == 0.20 and not st.passes(6, now)                             # an hour with no signals: p = price = zero edge = Kelly sizes to nothing
    assert "TRADE" in st.explain(5, now) and "no trade" in st.explain(6, now)
    # not enough signals in the window: no trade even though the few it has are great
    few = TrailingEVStats.from_signals(hist([1] * 5), ((14, 0.10),), 8, 0.02, "Kalshi")
    assert not few.passes(5, t0 + 5 * (day // 4) + 60) and "needs n>=8" in few.explain(5, t0 + 5 * (day // 4) + 60)
    # two rules: BOTH must hold.  100 losing signals 15-27 days ago and 12 winners in the last 13 days: the 14 d window is hot, the 28 d window (12 wins in 112) is not
    ref = t0 + 60 * day
    rows2 = [dict(leg="DOG", hour=5, open_ts=int(ref - 27 * day + i * (12 * day / 100)) // 900 * 900, win=0, cost=0.15) for i in range(100)]
    rows2 += [dict(leg="DOG", hour=5, open_ts=int(ref - 13 * day + i * day) // 900 * 900, win=1, cost=0.15) for i in range(12)]
    new = pd.DataFrame(rows2).assign(close_ts=lambda d: d.open_ts + 900)
    both = TrailingEVStats.from_signals(new, ((14, 0.10), (28, 0.05)), 8, 0.02, "Kalshi")
    only14 = TrailingEVStats.from_signals(new, ((14, 0.10),), 8, 0.02, "Kalshi")
    only28 = TrailingEVStats.from_signals(new, ((28, 0.05),), 8, 0.02, "Kalshi")
    n14, ev14, g14 = both.window(5, ref, 14); n28, ev28, g28 = both.window(5, ref, 28)
    assert n14 == 12 and ev14 > 0.10 and n28 == 112 and ev28 < 0.05
    assert only14.passes(5, ref) and not only28.passes(5, ref) and not both.passes(5, ref)           # all rules must hold
    assert "28 d EV" in both.explain(5, ref) and "no trade" in both.explain(5, ref)
    # a signal that settles while trading counts for the next decision; one the history file already has is ignored
    live = TrailingEVStats.from_signals(sig, ((14, 0.10),), 8, 0.02, "Kalshi")
    last_open = int(sig.open_ts.max())
    live.add_live("DOG", 5, last_open, True, 0.15)                                                   # already in the history: ignored
    assert live.window(5, now, 14)[0] == 16
    live.add_live("DOG", 5, last_open + day // 4, False, 0.15)
    assert live.window(5, now + day, 14)[0] == 17
    for bad in ((), ):
        try:
            TrailingEVStats(bad)
            assert False
        except ValueError:
            pass


def test_ev_filter_presets_replay_the_live_engine_exactly_and_only_trade_passing_hours():
    from trader.presets import find_preset, preset_config
    from trader.replay import verify
    rows = _hourly_history(60, 5)
    for name in ("FILTER-14", "FILTER-28", "FILTER-BOTH"):
        cfg = preset_config(name, date(2026, 6, 1), date(2026, 6, 28), capital=1_000.0)
        assert cfg.ev_rules and cfg.auto is None
        cmp, bt, rp = verify(rows, cfg)
        assert cmp.identical and len(bt.trades) >= 20, (name, cmp.text(), len(bt.trades))
        hrs = pd.to_datetime(bt.trades.entry_ts, unit="s", utc=True).dt.hour
        assert (hrs <= 5).mean() >= 0.9, (name, hrs.value_counts().to_dict())                       # the hot hours (0-5) are where it trades
    assert preset_config("FILTER-14", date(2026, 6, 1), date(2026, 6, 2)).ev_rules == ((14, 0.10),)
    assert preset_config("FILTER-28", date(2026, 6, 1), date(2026, 6, 2)).ev_rules == ((28, 0.05),)
    assert preset_config("FILTER-BOTH", date(2026, 6, 1), date(2026, 6, 2)).ev_rules == ((14, 0.10), (28, 0.05))
    assert find_preset("filter-28").startswith("FILTER-28") and preset_config("R2+", date(2026, 6, 1), date(2026, 6, 2)).ev_rules == ()
    # the live config round-trips through the ledger's JSON (a later reconcile rebuilds the filter)
    import json
    from trader.live import auto_active, config_from_dict, config_to_dict
    from trader.replay import live_config_from
    lc = live_config_from(preset_config("FILTER-BOTH", date(2026, 6, 1), date(2026, 6, 28)))
    assert config_from_dict(json.loads(json.dumps(config_to_dict(lc)))) == lc and lc.ev_rules == ((14, 0.10), (28, 0.05)) and not auto_active(lc)


def test_ev_filter_presets_replay_on_real_history():
    from trader.data import DATA, load_rows
    from trader.presets import preset_config
    from trader.replay import verify
    if not (DATA / "markets.csv").exists() or not (DATA / "candles.csv").exists():
        print("     (no local history - skipped)")
        return
    rows = load_rows()
    if rows.open_ts.max() < int(datetime(2026, 9, 30, tzinfo=timezone.utc).timestamp()):
        print("     (history too short - skipped)")
        return
    for name in ("FILTER-14", "FILTER-28", "FILTER-BOTH"):
        cmp, bt, rp = verify(rows, preset_config(name, date(2026, 8, 20), date(2026, 9, 30), capital=1_000.0))
        assert cmp.identical and len(bt.trades) >= 5, (name, cmp.text(), len(bt.trades))



# ── PROJ-EV: any hour while its projected EV is positive ──────────────────────────
def _flip_history(days, flip_day, seed, hour=3, p_good=0.60, p_bad=0.02, p_other=0.15):
    """Every window of every day with a 15c underdog.  ``hour`` is genuinely good (+40c) until ``flip_day`` and then genuinely bad; every other hour wins about its price (-3c after costs)."""
    rng = np.random.default_rng(seed)
    base = int(datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp())
    recs = []
    for d in range(days):
        for w in range(96):
            o = base + d * 86400 + w * 900
            hr = (w * 900) // 3600
            p = (p_good if d < flip_day else p_bad) if hr == hour else p_other
            y = int(rng.random() < p)
            for m in (12, 13):
                recs.append(dict(ticker=f"KXBTC15M-{o}", open_ts=o, minute=m, bid=0.14, ask=0.15, yes=y, vol=5_000.0))
    return pd.DataFrame(recs), base


def test_proj_ev_trades_an_hour_once_its_projection_turns_positive_and_stops_when_it_turns_negative():
    from trader import radar
    from trader.presets import preset_config
    from trader.replay import verify
    rows, base = _flip_history(75, 40, 7)
    cfg = preset_config("PROJ-EV", date(2026, 6, 1), date(2026, 7, 14), capital=1_000.0)
    assert cfg.min_edge == 0.0 and cfg.p_model == "by_hour" and len(cfg.params.hours) == 24 and cfg.auto is None and not cfg.ev_rules
    cmp, bt, rp = verify(rows, cfg)
    assert cmp.identical, cmp.text()                                                                   # live engine == backtest for this preset too
    t = bt.trades.copy()
    t["hour"] = pd.to_datetime(t.entry_ts, unit="s", utc=True).dt.hour
    t["day"] = (t.entry_ts - base) // 86400
    h3 = t[t.hour == 3]
    assert len(h3[h3.day < 40]) >= 10, h3.day.tolist()                                                  # it found the good hour and traded it before the flip ...
    assert h3[h3.day >= 40 + 21].empty, h3[h3.day >= 61].day.tolist()                                   # ... and had stopped trading it within three weeks of the flip (14 d memory)
    assert (t.p_used > t.quoted).all()                                                                  # every trade had a positive projected edge before costs: the hurdle is 'above zero'
    # the radar agrees: the hour is TRADE while its projection is positive and cold afterwards
    rc = radar.RadarConfig(min_edge=0.0, hour_k=100.0, halflife_days=14.0)
    now_good = float(base + 38 * 86400)
    now_bad = float(base + 72 * 86400)
    st_good = radar.hour_radar(rows, now_good, rc, ())
    st_bad = radar.hour_radar(rows, now_bad, rc, ())
    assert st_good[st_good.hour == 3].iloc[0].status == "TRADE" and st_bad[st_bad.hour == 3].iloc[0].status != "TRADE"
    assert st_good[st_good.hour == 3].iloc[0].est_edge > 0 > st_bad[st_bad.hour == 3].iloc[0].est_edge


def test_proj_ev_replays_the_live_engine_on_real_history():
    from trader.data import DATA, load_rows
    from trader.presets import preset_config
    from trader.replay import verify
    if not (DATA / "markets.csv").exists() or not (DATA / "candles.csv").exists():
        print("     (no local history - skipped)")
        return
    rows = load_rows()
    if rows.open_ts.max() < int(datetime(2026, 9, 30, tzinfo=timezone.utc).timestamp()):
        print("     (history too short - skipped)")
        return
    cmp, bt, rp = verify(rows, preset_config("PROJ-EV", date(2026, 8, 20), date(2026, 9, 30), capital=1_000.0))
    assert cmp.identical and len(bt.trades) >= 20, (cmp.text(), len(bt.trades))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("all trader tests passed")
