"""Offline tests for trader/preflight.py (no real network, no real credentials; a loopback mock exchange for the account checks).
Run:  python test_preflight.py
"""
import tempfile
import threading
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import test_trader as tt
from trader import kalshi
from trader import preflight as pf
from trader.live import LiveConfig
from trader.strategy import StrategyParams

UTC = timezone.utc
TS = lambda *a: datetime(*a, tzinfo=UTC).timestamp()
NOW = TS(2026, 10, 20, 21, 46)                               # the evening start
R2 = StrategyParams(hours=(22, 23), cost_min=0.15, cost_max=0.35)


def make_rows(win_fn, start=(2026, 8, 20), end=(2026, 10, 20, 21, 45), price=0.20):
    """A quote at minute 12/13 for every 15-minute window in [start, end]; the underdog is YES at ``price``; yes = win_fn(open_ts)."""
    t0, t1 = int(TS(*start)), int(TS(*end))
    ots = np.arange(t0 // 900 * 900, t1, 900)
    rows = []
    for m in (12, 13):
        rows.append(pd.DataFrame(dict(ticker=[f"W{o}" for o in ots], open_ts=ots, minute=m, bid=price - 0.01, ask=price, yes=[int(win_fn(int(o))) for o in ots])))
    return pd.concat(rows).sort_values(["open_ts", "minute"]).reset_index(drop=True)


def rng_win(p, seed=1):
    rng = np.random.default_rng(seed)
    cache = {}
    return lambda o: cache.setdefault(o, rng.random() < p)


class Q:
    def window_ticker(self, w): return f"KXBTC15M-{w}"
    def quote(self, t): return (0.20, 0.21)


class NoQuote(Q):
    def quote(self, t): return None


class Resp:
    def __init__(self, code=200, headers=None, j=None):
        self.status_code, self.headers, self._j = code, headers or {}, j or {"exchange_active": True, "trading_active": True}
    def json(self): return self._j


def ok_get(*a, **k): return Resp()


def down_get(*a, **k): raise requests.ConnectionError("boom")


def cfg(**kw):
    base = dict(capital=1000.0, params=R2, lookback_days=45)
    base.update(kw)
    return LiveConfig(**base)


def run(rows, c=None, live=False, tmp=None, **kw):
    kw.setdefault("refresh", False)
    kw.setdefault("quotes", Q())
    kw.setdefault("http_get", ok_get)
    kw.setdefault("coverage_fn", lambda *a, **k: (10, 10))
    kw.setdefault("now", NOW)
    return pf.run_preflight(c or cfg(), live=live, rows=rows, log=lambda s: None, state_directory=Path(tmp or tempfile.mkdtemp()), **kw)


def by_name(rep, name):
    return [c for c in rep.checks if c.name == name]


# ── forward test ──────────────────────────────────────────────────────────────
def test_forward_sprt_accept_reject_and_sticky():
    r = pf.evaluate_forward(make_rows(lambda o: True))
    assert r.verdict == "ACCEPT" and r.n > 60 and r.decided_at is not None and r.ev_c > 70
    # first 40 forward windows lose, then everything wins: the reject boundary was crossed first and is terminal
    lose_until = TS(2026, 10, 8) + 5 * 86400
    r = pf.evaluate_forward(make_rows(lambda o: o >= lose_until))
    assert r.verdict == "REJECT" and r.decided_at is not None and r.decided_at < lose_until + 86400 and r.llr > 0
    r = pf.evaluate_forward(make_rows(lambda o: False, end=(2026, 10, 9, 23, 45)))      # ~1.3 days of losses: not enough evidence either way
    assert r.verdict == "CONTINUE" and r.llr < 0
    assert pf.evaluate_forward(make_rows(lambda o: True, end=(2026, 10, 7, 23, 0))).n == 0


def test_forward_matches_forward_test_script_on_real_data():
    try:
        import forward_test
        from trader.data import load_rows
        rows = load_rows()
    except Exception as e:                                       # noqa: BLE001
        print("   (skipped: real data/forward_test not importable:", e, ")")
        return
    T = forward_test.evaluate()
    r = pf.evaluate_forward(rows)
    assert r.n == len(T), (r.n, len(T))
    assert abs(r.cum_c - T.pnl.sum() * 100) < 1e-6, (r.cum_c, T.pnl.sum() * 100)


def test_forward_needs_run_detection():
    rows = make_rows(lambda o: True)
    need, why = pf.forward_needs_run({}, rows, NOW)
    assert need and "first check" in why
    hi = int(rows.open_ts.max()) + 900
    st = {"forward": {"ts": NOW - 600, "data_hi": hi}}
    need, why = pf.forward_needs_run(st, rows, NOW)
    assert not need and "up to date" in why
    st = {"forward": {"ts": NOW - 600, "data_hi": hi - 4 * 900}}
    need, why = pf.forward_needs_run(st, rows, NOW)
    assert need and "4 new windows" in why
    st = {"forward": {"ts": NOW - 13 * 3600, "data_hi": hi}}
    assert pf.forward_needs_run(st, rows, NOW)[0]


# ── verdicts ──────────────────────────────────────────────────────────────────
def test_go_when_everything_passes():
    rep = run(make_rows(lambda o: True))
    assert rep.forward.verdict == "ACCEPT"
    assert rep.verdict == "GO", rep.text()
    assert rep.stake_cap is None
    assert by_name(rep, "Learner gate")[0].status == pf.PASS and "OPEN" in by_name(rep, "Learner gate")[0].detail
    assert by_name(rep, "Kalshi account")[0].status == pf.SKIP


def test_caution_while_forward_test_unconfirmed_caps_stake():
    # a real-looking edge in the trailing window but only ~1 day of forward trades: evidence not confirmed yet
    rows = make_rows(lambda o: o < TS(2026, 10, 8) or np.random.default_rng(int(o)).random() < 0.45, end=(2026, 10, 9, 23, 45))
    rep = run(rows, now=TS(2026, 10, 10, 21, 46))
    assert rep.forward.verdict == "CONTINUE"
    f = by_name(rep, "Forward test (pre-registered R1)")[0]
    assert f.status == pf.WARN and f.cap == pf.CAUTION_CAP
    assert rep.verdict == "CAUTION" and rep.stake_cap == pf.CAUTION_CAP
    assert "0.5%" in rep.headline()


def test_no_go_when_forward_test_rejects():
    lose_until = TS(2026, 10, 8) + 6 * 86400
    rep = run(make_rows(lambda o: o >= lose_until or o < TS(2026, 10, 8)))
    assert rep.forward.verdict == "REJECT" and rep.verdict == "NO-GO"
    assert any("Forward test" in c.name for c in rep.of(pf.FAIL))


def test_no_go_when_trailing_edge_negative():
    rep = run(make_rows(lambda o: o < TS(2026, 9, 1)))                       # wins before September, nothing for 7 weeks
    c = by_name(rep, "Trailing edge (your preset)")[0]
    assert c.status == pf.FAIL and rep.verdict == "NO-GO"


def test_gate_closed_is_a_warning_not_a_failure():
    # win rate equals the price (20c): no edge -> the learner never opens
    rep = run(make_rows(rng_win(0.20, seed=3)))
    g = by_name(rep, "Learner gate")[0]
    assert g.status == pf.WARN and "CLOSED" in g.detail


def test_stale_data_updates_when_allowed_and_fails_when_not():
    fresh = make_rows(lambda o: True)
    old = fresh[fresh.open_ts < TS(2026, 10, 19, 12)].copy()
    calls = []
    rep = run(old, refresh=True, updater=lambda log: calls.append(1) or True, loader=lambda: fresh)
    assert calls and rep.updated_data and by_name(rep, "Local history")[0].status == pf.PASS
    rep = run(old, refresh=False)                                            # 33.7 h behind and not allowed to update
    assert by_name(rep, "Local history")[0].status == pf.WARN
    older = fresh[fresh.open_ts < TS(2026, 10, 18, 12)]
    rep = run(older, refresh=False)
    assert by_name(rep, "Local history")[0].status == pf.FAIL and rep.verdict == "NO-GO"
    # updater blows up: the check carries on with what is on disk and says so
    rep = run(old, refresh=True, updater=lambda log: (_ for _ in ()).throw(RuntimeError("net down")), loader=lambda: fresh)
    assert any("update failed" in n for n in rep.notes) and not rep.updated_data


def test_update_is_not_repeated_within_20_minutes():
    fresh = make_rows(lambda o: True)
    old = fresh[fresh.open_ts < TS(2026, 10, 20, 19, 0)].copy()                 # 2.8 h behind: Kalshi's normal candle lag
    calls = []
    d = tempfile.mkdtemp()
    kw = dict(refresh=True, updater=lambda log: calls.append(1) or True, loader=lambda: old, tmp=d)
    run(old, **kw)
    run(old, now=NOW + 600, **kw)
    assert len(calls) == 1
    run(old, now=NOW + 25 * 60, **kw)
    assert len(calls) == 2


def test_feed_failures_block():
    rows = make_rows(lambda o: True)
    rep = run(rows, http_get=down_get)
    assert by_name(rep, "Kalshi exchange")[0].status == pf.FAIL and rep.verdict == "NO-GO"
    rep = run(rows, quotes=NoQuote())
    assert by_name(rep, "Live quote")[0].status == pf.FAIL
    rep = run(rows, http_get=lambda *a, **k: Resp(headers={"Date": "Tue, 20 Oct 2026 21:50:00 GMT"}))           # 4 min ahead of NOW
    assert by_name(rep, "PC clock")[0].status == pf.FAIL
    rep = run(rows, http_get=lambda *a, **k: Resp(j={"exchange_active": True, "trading_active": False}))
    assert by_name(rep, "Kalshi exchange")[0].status == pf.FAIL


def test_clock_skew_uses_request_time_not_start_time():
    from email.utils import format_datetime
    hdr = {"Date": format_datetime(datetime.fromtimestamp(NOW, UTC), usegmt=True)}
    # the pre-flight "started" 30 s ago (a long data update in between) but the server's clock agrees with the PC's clock right now
    out = pf.kalshi_checks(NOW - 30, Q(), lambda *a, **k: Resp(headers=hdr), clock=lambda: NOW)
    assert {c.name: c for c in out}["PC clock"].status == pf.PASS
    hdr = {"Date": format_datetime(datetime.fromtimestamp(NOW + 12, UTC), usegmt=True)}      # genuinely 12 s off
    out = pf.kalshi_checks(NOW, Q(), lambda *a, **k: Resp(headers=hdr), clock=lambda: NOW)
    assert {c.name: c for c in out}["PC clock"].status == pf.WARN


def test_config_checks():
    rows = make_rows(lambda o: True)
    rep = run(rows, c=cfg(max_frac=0.10))
    assert by_name(rep, "Sizing")[0].status == pf.FAIL and rep.verdict == "NO-GO"
    rep = run(rows, c=cfg(max_frac=0.03))
    assert by_name(rep, "Sizing")[0].status == pf.WARN
    rep = run(rows, c=cfg(params=StrategyParams(hours=tuple(range(24)), legs=("DOG", "FAV"))))
    p = by_name(rep, "Preset")[0]
    assert p.status == pf.WARN and p.cap == pf.CAUTION_CAP
    rep = run(rows, c=cfg(prob_source="manual", manual_p=0.5))
    assert by_name(rep, "Learner gate")[0].status == pf.WARN


def test_core_scout_preset_is_judged_on_its_core_hours_and_radar_warns_on_a_move():
    all_hours = StrategyParams(hours=tuple(range(24)), cost_min=0.15, cost_max=0.35)
    c = cfg(params=all_hours, p_model="by_hour", hour_k=100, core_hours=(22, 23), core_k=10, halflife_days=14.0)
    rep = run(make_rows(lambda o: True), c=c)
    assert by_name(rep, "Preset")[0].status == pf.PASS and "core hours 22,23" in by_name(rep, "Preset")[0].detail
    assert by_name(rep, "Trailing edge (your preset)")[0].status == pf.PASS and rep.verdict == "GO", rep.text()
    assert by_name(rep, "Regime radar (all 24 hours)")[0].status == pf.PASS
    # the edge sits in hour 5 instead of the core hours: core hours are dead (FAIL) and the radar says it moved
    hot5 = lambda o: (o >= TS(2026, 9, 1)) and int((o % 86400) // 3600) == 5 or (o < TS(2026, 9, 1) and int((o % 86400) // 3600) in (22, 23))
    rep = run(make_rows(hot5), c=c)
    r = by_name(rep, "Regime radar (all 24 hours)")[0]
    assert r.status == pf.WARN and "moved" in r.detail and by_name(rep, "Trailing edge (your preset)")[0].status == pf.FAIL
    # without core hours the all-hours preset is outside the tested family
    rep = run(make_rows(lambda o: True), c=cfg(params=all_hours, p_model="by_hour"))
    assert by_name(rep, "Preset")[0].status == pf.WARN


def test_previous_session_and_state_roundtrip():
    d = Path(tempfile.mkdtemp())
    pf.record_session_end("KILLED", "live", -35.0, 4, d=d, now=NOW - 3600)
    st = pf.load_state(d)
    assert st["last_session"]["why"] == "KILLED"
    c = pf.previous_session_check(st, NOW)
    assert c.status == pf.WARN and "safety mechanism" in c.detail
    pf.record_session_end("stopped by user", "paper", 12.0, 3, d=d, now=NOW - 3600)
    assert pf.previous_session_check(pf.load_state(d), NOW).status == pf.PASS
    # forward bookkeeping is persisted by a run
    run(make_rows(lambda o: True), tmp=str(d))
    assert pf.load_state(d)["forward"]["verdict"] == "ACCEPT"


def test_timing():
    c = pf.timing_check(cfg(), NOW)                                          # 21:46 -> next decision 22:12
    assert c.status == pf.PASS and "22:12:00" in c.detail and "26 min" in c.detail
    assert pf.next_decision(TS(2026, 10, 20, 22, 30), R2) == TS(2026, 10, 20, 22, 42)
    assert pf.next_decision(TS(2026, 10, 20, 23, 59, 0), R2) == TS(2026, 10, 21, 22, 12)
    assert pf.timing_check(cfg(), TS(2026, 10, 20, 3, 0)).status == pf.INFO


# ── account checks against the signature-verifying loopback mock ───────────────
class QTicks(Q):
    """A quote source that can describe the live market (for the price-tick check)."""
    def __init__(self, ranges, status="active"): self.ranges, self.status = ranges, status
    def market_info(self, t): return {"ticker": t, "status": self.status, "price_ranges": self.ranges}


def test_account_checks_against_mock_exchange():
    srv, url = tt._mock_server()
    try:
        client = tt._client(url)
        rows = make_rows(lambda o: True)
        rep = run(rows, c=cfg(capital=900.0, mode="live"), live=True, client=client)
        acct = {c.name: c for c in rep.checks if c.group == "account"}
        assert acct["Buying power"].status == pf.PASS and acct["Order price ticks"].status in (pf.PASS, pf.WARN, pf.INFO), rep.text()
        assert acct["Resting orders"].status == pf.PASS and acct["Open positions"].status == pf.PASS
        assert acct["Clock vs Kalshi"].status == pf.PASS and rep.client is client
        # not enough buying power: warn at >=50%, fail below
        rep = run(rows, c=cfg(capital=1500.0, mode="live"), live=True, client=client)
        assert {c.name: c for c in rep.checks}["Buying power"].status == pf.WARN
        rep = run(rows, c=cfg(capital=5000.0, mode="live"), live=True, client=client)
        assert {c.name: c for c in rep.checks}["Buying power"].status == pf.FAIL and rep.verdict == "NO-GO"
        # leftovers from an earlier session
        tt.MockKalshi.positions = [{"ticker": "KXBTC15M-26OCT192300-00", "position_fp": "5.00"}]
        tt.MockKalshi.resting = [{"order_id": "x"}]
        rep = run(rows, c=cfg(capital=900.0, mode="live"), live=True, client=client)
        acct = {c.name: c for c in rep.checks if c.group == "account"}
        assert acct["Open positions"].status == pf.WARN and acct["Resting orders"].status == pf.WARN and rep.verdict == "CAUTION"
        # wrong key -> 401 -> blocked, and no secret in the report text
        other = tt.rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(tt.serialization.Encoding.PEM, tt.serialization.PrivateFormat.PKCS8, tt.serialization.NoEncryption()).decode()
        bad = tt._client(url, kalshi.Credentials(tt.TEST_KEY_ID, other))
        rep = run(rows, c=cfg(capital=900.0, mode="live"), live=True, client=bad)
        assert rep.verdict == "NO-GO" and tt.TEST_PEM not in rep.text() and other not in rep.text()
        # no key path and no client
        rep = run(rows, c=cfg(mode="live"), live=True)
        assert by_name(rep, "Account access")[0].status == pf.FAIL
    finally:
        srv.shutdown()


def test_order_price_tick_check():
    srv, url = tt._mock_server()
    try:
        client = tt._client(url)
        rows = make_rows(lambda o: True)
        tapered = [{"start": "0.0000", "end": "0.1000", "step": "0.0010"}, {"start": "0.1000", "end": "0.9000", "step": "0.0100"}, {"start": "0.9000", "end": "1.0000", "step": "0.0010"}]
        def tick(q):
            rep = run(rows, c=cfg(capital=900.0, mode="live"), live=True, client=client, quotes=q)
            return {c.name: c for c in rep.checks}["Order price ticks"]
        assert tick(QTicks(tapered)).status == pf.PASS                                               # 0.1c / 1c steps: whole-cent limit prices are always valid
        assert tick(QTicks([{"start": "0.0", "end": "1.0", "step": "0.05"}])).status == pf.WARN      # 5c steps: cent-rounded limits could be rejected
        assert tick(QTicks(tapered, status="closed")).status == pf.WARN
    finally:
        srv.shutdown()


def test_report_text_and_save():
    rep = run(make_rows(lambda o: True))
    txt = rep.text()
    assert "VERDICT FOR LIVE TRADING" in txt and "[PASS]" in txt and "Forward test" in txt
    p = pf.save_report(rep, Path(tempfile.mkdtemp()))
    assert p and p.read_text(encoding="utf-8") == txt


def test_one_crashing_check_does_not_hide_the_rest():
    rep = run(make_rows(lambda o: True), coverage_fn=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    names = [c.name for c in rep.checks]
    assert "Learner gate" in names and "Kalshi exchange" in names



def test_buying_power_check_counts_only_the_cash_on_the_shard_the_orders_draw_on():
    """The pre-flight of 2026-10-12 said 'buying power $1,004.33 covers the capital' while the cash was on the wrong exchange shard - every order was then rejected."""
    srv, url = tt._mock_server()
    try:
        client = tt._client(url)
        rows = make_rows(lambda o: True)
        tt.MockKalshi.shards, tt.MockKalshi.series_index = {0: 1004.33, 2: 0.0}, 2
        rep = run(rows, c=cfg(capital=1000.0, mode="live"), live=True, client=client, quotes=kalshi.KalshiQuotes(base=client.base))
        bp = by_name(rep, "Buying power")[0]
        assert bp.status == pf.FAIL and "NO CASH" in bp.detail and "shard 2" in bp.detail and "Crypto & Commodities" in bp.detail and "$1,004.33" in bp.detail and "Exchange Indexes" in bp.detail, bp.detail
        assert rep.verdict == "NO-GO"
        tt.MockKalshi.shards = {0: 1004.33, 2: 600.0}                                           # partly funded: a warning, stakes are capped to the shard's cash
        bp = by_name(run(rows, c=cfg(capital=1000.0, mode="live"), live=True, client=client, quotes=kalshi.KalshiQuotes(base=client.base)), "Buying power")[0]
        assert bp.status == pf.WARN and "shard 2" in bp.detail and "$600.00" in bp.detail, bp.detail
        tt.MockKalshi.shards = {0: 4.33, 2: 1000.0}                                             # funded where it counts
        bp = by_name(run(rows, c=cfg(capital=1000.0, mode="live"), live=True, client=client, quotes=kalshi.KalshiQuotes(base=client.base)), "Buying power")[0]
        assert bp.status == pf.PASS and "shard 2" in bp.detail, bp.detail
    finally:
        srv.shutdown()


def test_buying_power_check_with_auto_fund_counts_cash_that_will_be_moved():
    srv, url = tt._mock_server()
    try:
        client = tt._client(url)
        rows = make_rows(lambda o: True)
        tt.MockKalshi.shards, tt.MockKalshi.series_index = {0: 1000.0, 2: 4.33}, 2                     # the user's account on 2026-10-12
        q = kalshi.KalshiQuotes(base=client.base)
        off = by_name(run(rows, c=cfg(capital=1000.0, mode="live"), live=True, client=client, quotes=q), "Buying power")[0]
        assert off.status == pf.FAIL and "shard 2" in off.detail
        on = by_name(run(rows, c=cfg(capital=1000.0, mode="live", auto_fund=True), live=True, client=client, quotes=q), "Buying power")[0]
        assert on.status == pf.PASS and "AUTO-FUND is on" in on.detail and "$995.67" in on.detail and "own account" in on.detail, on.detail
        tot_short = by_name(run(rows, c=cfg(capital=5000.0, mode="live", auto_fund=True), live=True, client=client, quotes=q), "Buying power")[0]
        assert tot_short.status == pf.WARN and "AUTO-FUND" in tot_short.detail                         # not enough cash in the whole account for the capital: warned, stakes follow the cash
        cap = by_name(run(rows, c=cfg(capital=1000.0, mode="live", auto_fund=True, fund_max=100.0), live=True, client=client, quotes=q), "Buying power")[0]
        assert cap.status == pf.PASS and "$95.67" in cap.detail, cap.detail                          # fund_max 100: moves only up to $100 on the shard
    finally:
        srv.shutdown()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("all preflight tests passed")
