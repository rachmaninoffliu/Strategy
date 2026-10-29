"""
Live / paper trading engine for the stat-arb book on Alpaca (no Qt).

What it does, once per session, after the close:
    refresh data  ->  signals for the last complete session  ->  Book.decide (the SAME function the backtest runs)  ->  target shares  ->  order list vs the engine's ledger
    ->  (only when ARMED) market-on-open orders queued for the next open

Safety, in the same spirit as the Supertrend app:
  * starts on PAPER and DRY RUN; ``armed`` has to be switched on explicitly, and a live (real-money) account additionally needs the typed confirmation the GUI asks for
  * orders are tagged ``SA-<yyyymmdd>-<symbol>``; the engine only ever touches symbols in its own ledger or its own plan, never the rest of the account
  * shorts are only planned in names Alpaca marks shortable AND easy-to-borrow; whole shares only; a minimum ticket size; a per-name cap; a gross cap (Reg-T 2x overnight)
  * a daily-loss halt and a KILL switch (cancel the engine's orders and flatten the engine's ledger)
  * OPG orders are accepted by Alpaca between 19:00 ET and 09:28 ET only: the plan is made after the close, the submit window is the evening / early morning
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, time as dtime
from typing import Callable, Optional

import numpy as np
import pandas as pd

import statarb_data as D
from statarb_core import FactorCfg, PortCfg, TD, build_signals, shock_table
from statarb_sim import DayInputs, Portfolio, cost_vec, sector_etf_index, shock_multiplier

NY = "America/New_York"
PREFIX = "SA"


@dataclass
class LiveSettings:
    paper: bool = True
    armed: bool = False                          # False = DRY RUN: plans are made and shown, nothing is sent
    universe: str = "core"                       # 'core' (curated large caps) | 'broad' (survivorship-free)
    equity_fraction: float = 1.0                 # share of the account equity this strategy may use
    capital_override: float = 0.0                # > 0: use this much instead of account equity x fraction
    exec_mode: str = "opg"                       # 'opg' = market-on-open queued | 'day' = plain market order now (testing)
    min_trade_usd: float = 250.0
    max_name_pct: float = 0.05                   # of deployed capital in one name
    max_orders: int = 250
    daily_loss_halt: float = 0.025
    require_etb: bool = True
    dd_half: float = 0.08                        # drawdown from the high-water mark: halve the gross ...
    dd_flat: float = 0.10                        # ... and go flat (until the base is reset by hand)


@dataclass
class Check:
    name: str
    level: str                                   # ok | warn | fail
    msg: str


@dataclass
class Plan:
    decision_date: pd.Timestamp
    deploy: float
    equity: float
    g_target: float
    shock_mult: float
    n_long: int
    n_short: int
    orders: pd.DataFrame
    targets: dict
    exposures: dict
    regime: dict
    notes: list
    book_after: dict
    ledger_after: dict
    hedge: dict = field(default_factory=dict)
    vol_state: dict = field(default_factory=dict)
    hwm: float = 0.0


def state_path(paper: bool, universe: str) -> str:
    os.makedirs(D.REPORTS, exist_ok=True)
    return os.path.join(D.REPORTS, f"statarb_state_{'paper' if paper else 'live'}_{universe}.json")


def clock_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NY)


def last_complete_session(panel, now: Optional[pd.Timestamp] = None) -> int:
    """Index of the newest session whose daily bar is final: today's bar only counts once it is after 16:20 ET (the SIP bar is complete a few minutes after the close)."""
    now = now or clock_now()
    i = len(panel.dates) - 1
    if panel.dates[i].date() == now.date() and (now.hour, now.minute) < (16, 20):
        i -= 1
    return i


def expected_last_session(now: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """The newest NYSE session whose daily bar should be final by now (holidays via the exchange calendar)."""
    now = now or clock_now()
    today = now.tz_localize(None).normalize()
    try:
        import pandas_market_calendars as mcal
        days = mcal.get_calendar("NYSE").valid_days(start_date=today - pd.Timedelta(days=12), end_date=today).tz_localize(None).normalize()
    except Exception:                                                                  # noqa: BLE001
        days = pd.bdate_range(today - pd.Timedelta(days=12), today)
    days = [d for d in days if d <= today]
    if days and days[-1] == today and (now.hour, now.minute) < (16, 20):
        days = days[:-1]
    return pd.Timestamp(days[-1])


def opg_window_open(now: Optional[pd.Timestamp] = None) -> tuple[bool, str]:
    """Alpaca accepts market-on-open orders from 19:00 ET to 09:28 ET (weekends included for Monday's open)."""
    now = now or clock_now()
    t = (now.hour, now.minute)
    if t >= (19, 0) or t < (9, 28):
        return True, "OPG window open (19:00 - 09:28 ET)"
    return False, f"OPG window closed (it is {now:%H:%M} ET; opens 19:00 ET)"


class LiveEngine:
    def __init__(self, fcfg: FactorCfg, pcfg: PortCfg, live: LiveSettings, log: Callable[[str, str], None] = lambda m, k="info": print(m)):
        self.f, self.p, self.live, self.log = fcfg, pcfg, live, log
        self.tc = None
        self.panel: Optional[D.Panel] = None
        self.earn = None
        self.acct: dict = {}

    # ── broker ───────────────────────────────────────────────────────────────────
    def connect(self) -> dict:
        from alpaca.trading.client import TradingClient
        k, s = D.creds(self.live.paper)
        self.tc = TradingClient(k, s, paper=self.live.paper)
        t0 = time.time()
        a = self.tc.get_account()
        clk = self.tc.get_clock()
        self.acct = dict(paper=self.live.paper, number=str(a.account_number)[:4] + "****", status=str(getattr(a.status, "value", a.status)), equity=float(a.equity), cash=float(a.cash),
                         buying_power=float(a.buying_power), multiplier=str(a.multiplier), shorting=bool(a.shorting_enabled), blocked=bool(a.trading_blocked or a.account_blocked),
                         pattern_day_trader=bool(a.pattern_day_trader), market_open=bool(clk.is_open), next_open=pd.Timestamp(clk.next_open), next_close=pd.Timestamp(clk.next_close),
                         latency_ms=int((time.time() - t0) * 1000), last_equity=float(a.last_equity))
        return self.acct

    def loss_halt(self) -> tuple[bool, str]:
        """True when the account is down more than ``daily_loss_halt`` since the previous close (equity vs Alpaca's last_equity): no new risk, closing orders only."""
        a = self.acct or {}
        eq, last = a.get("equity"), a.get("last_equity")
        if not eq or not last or last <= 0:
            return False, ""
        r = eq / last - 1.0
        if self.live.daily_loss_halt > 0 and r <= -self.live.daily_loss_halt:
            return True, f"account is {r:.2%} on the day (limit -{self.live.daily_loss_halt:.1%})"
        return False, ""

    def positions(self) -> dict:
        return {p.symbol: float(p.qty) for p in self.tc.get_all_positions()}

    # ── state ────────────────────────────────────────────────────────────────────
    def load_state(self) -> dict:
        path = state_path(self.live.paper, self.live.universe)
        if os.path.exists(path):
            try:
                return json.load(open(path, encoding="utf-8"))
            except (OSError, ValueError) as e:
                raise RuntimeError(f"state file {path} is unreadable ({e}); fix or move it - starting from an empty ledger would forget real positions") from e
        return {"book": {}, "ledger": {}, "held_w": {}, "held_date": None, "var_u": None, "n_u": 0, "hwm": None, "created": str(pd.Timestamp.now())}

    def save_state(self, st: dict) -> None:
        path = state_path(self.live.paper, self.live.universe)
        tmp = path + ".tmp"
        json.dump(st, open(tmp, "w", encoding="utf-8"), indent=1, default=str)
        os.replace(tmp, path)

    # ── data ─────────────────────────────────────────────────────────────────────
    def load_panel(self, refresh: bool = False) -> D.Panel:
        if self.live.universe == "broad":
            import statarb_broad as B
            if refresh:
                B.update_broad_cache(log=lambda m: self.log(m, "info"))
            self.panel = B.load_broad_panel()
        else:
            self.panel = D.load_panel(refresh=refresh, log=lambda m: self.log(m, "info"))
        if refresh:
            try:
                self.earn = D.update_earnings(list(self.panel.stocks), log=lambda m: self.log(m, "info"))
            except Exception as e:                                                     # noqa: BLE001
                self.log(f"earnings calendar refresh failed ({type(e).__name__}: {str(e)[:80]}) - using the cached one", "warn")
                self.earn = D.load_earnings()
        else:
            self.earn = D.load_earnings()
        return self.panel

    # ── checks ───────────────────────────────────────────────────────────────────
    def preflight(self) -> list[Check]:
        out: list[Check] = []
        add = lambda n, l, m: out.append(Check(n, l, m))
        try:
            a = self.connect()
            add("broker", "ok", f"{'PAPER' if a['paper'] else 'LIVE'} account {a['number']} · {a['status']} · equity ${a['equity']:,.0f} · buying power ${a['buying_power']:,.0f} · {a['latency_ms']} ms")
            if a["blocked"]:
                add("account", "fail", "trading is blocked on this account")
            if not a["shorting"]:
                add("shorting", "fail", "shorting is not enabled on this account (needed for the long/short book)")
            mult = float(a["multiplier"] or 1)
            if mult < 2:
                add("margin", "warn", f"account multiplier is {mult:g}x - the book needs a margin account (2x) for its short leg")
            halted, why = self.loss_halt()
            if halted:
                add("loss halt", "fail", "DAILY-LOSS HALT ACTIVE: " + why)
            if a["equity"] < 25_000 and not a["paper"]:
                add("pdt", "warn", "equity < $25k: day-trading rules apply to a margin account (this strategy holds overnight, so it should not day-trade)")
        except Exception as e:                                                         # noqa: BLE001
            add("broker", "fail", f"cannot reach Alpaca: {type(e).__name__}: {str(e)[:120]}")
            return out
        try:
            if self.panel is None:
                self.load_panel(False)
            i = last_complete_session(self.panel)
            last = self.panel.dates[i]
            age = (clock_now().tz_localize(None).normalize() - last).days
            add("data", "ok" if age <= 3 else "warn", f"{self.panel.describe()} · last complete session {last:%Y-%m-%d} ({age} d ago)")
            ea = D.load_earnings()
            add("earnings", "ok" if len(ea) else "warn", f"earnings calendar: {ea.symbol.nunique() if len(ea) else 0} symbols" if len(ea) else "no earnings calendar - the earnings filters are inert")
        except Exception as e:                                                         # noqa: BLE001
            add("data", "fail", f"{type(e).__name__}: {str(e)[:140]}")
        ok, msg = opg_window_open()
        add("opg window", "ok" if ok else "warn", msg)
        st = self.load_state()
        add("state", "ok", f"ledger: {len(st.get('ledger', {}))} positions · book: {len(st.get('book', {}).get('positions', {}))} states")
        return out

    # ── planning ─────────────────────────────────────────────────────────────────
    def _update_vol_state(self, st: dict, panel, dec: int) -> None:
        """Advance the EWMA of the unit-gross daily return from the weights held since the last plan (open-to-open, same as the backtest; stocks and the ETF hedge)."""
        hd = st.get("held_date")
        if not hd or not st.get("held_w"):
            return
        d0 = int(panel.dates.searchsorted(pd.Timestamp(hd)))
        px_open = pd.concat([panel.open, panel.etf_open], axis=1)
        px_open = px_open.loc[:, ~px_open.columns.duplicated()]
        O = px_open.to_numpy(float)
        cols = {c: j for j, c in enumerate(px_open.columns)}
        alpha = 1.0 - 0.5 ** (1.0 / max(self.p.vol_halflife, 1))
        for d in range(d0 + 1, dec + 1):                                              # held from the open of d0+1 ... earns open-to-open returns until the open of dec+1 (not yet known)
            if d + 1 >= len(O):
                break
            gross = sum(abs(w) for w in st["held_w"].values())
            if gross < 0.2:
                continue
            pnl = 0.0
            for sym, w in st["held_w"].items():
                j = cols.get(sym)
                if j is not None and np.isfinite(O[d + 1, j]) and np.isfinite(O[d, j]) and O[d, j] > 0:
                    pnl += w * (O[d + 1, j] / O[d, j] - 1.0)
            u = pnl / gross
            st["var_u"] = u * u if st.get("var_u") is None else (1 - alpha) * st["var_u"] + alpha * u * u
            st["n_u"] = int(st.get("n_u", 0)) + 1

    def check_fresh(self, panel) -> None:
        """Refuse to plan on stale data: a stale stock panel next to fresher ETFs would otherwise look like 'every stock vanished' and plan to close the book."""
        dec = last_complete_session(panel)
        want = expected_last_session()
        have = panel.dates[dec]
        if have < want:
            raise RuntimeError(f"market data is stale: the newest complete session in the cache is {have:%Y-%m-%d} but {want:%Y-%m-%d} should be available - press 'Refresh data + plan'")
        px = panel.close.iloc[dec]
        prev = panel.close.iloc[max(dec - 1, 0)]
        if px.notna().sum() < 0.8 * prev.notna().sum():
            raise RuntimeError(f"only {int(px.notna().sum())} stocks have a price on {have:%Y-%m-%d} against {int(prev.notna().sum())} the day before - the stock cache is incomplete, refresh it")

    def plan(self, refresh: bool = False) -> Plan:
        if self.panel is None or refresh:
            self.load_panel(refresh)
        panel = self.panel
        self.check_fresh(panel)
        dec = last_complete_session(panel)
        dec_date = panel.dates[dec]
        st = self.load_state()
        cut = _truncate(panel, dec)
        start = (dec_date - pd.Timedelta(days=9)).strftime("%Y-%m-%d")
        self.log(f"signals for {dec_date:%Y-%m-%d} ({len(cut.stocks)} names) ...", "info")
        f = FactorCfg(**{**asdict(self.f), "start": start})
        sig = build_signals(cut, f, earnings=self.earn, n_jobs=1)
        i = len(cut.dates) - 1
        self._update_vol_state(st, cut, i)
        p = self.p
        notes: list = []
        unit_gross = max(p.w_rev * (1.0 if p.hedge != "etf" else 2.0) + p.w_eap * (2.0 if p.eap_hedge else 1.0), 1e-9)
        g_cap = p.g_max / max(unit_gross, 1.0)
        var_u, n_u = st.get("var_u"), int(st.get("n_u", 0))
        g = float(np.clip(p.vol_target / math.sqrt(var_u * TD), p.g_min, g_cap)) if (var_u and n_u >= p.vol_warmup) else min(p.g_init, g_cap)
        mult = float(shock_multiplier(sig.regime, p).iloc[i])
        g_t = float(min(g * mult, g_cap))
        equity = self.acct.get("equity", 0.0) if self.acct else 0.0
        hwm = max(float(st.get("hwm") or 0.0), float(equity))
        dd = equity / hwm - 1.0 if hwm > 0 and equity > 0 else 0.0
        if dd <= -self.live.dd_flat:
            g_t = 0.0
            notes.append(f"DRAWDOWN STOP: equity is {dd:.1%} from its high-water mark (limit -{self.live.dd_flat:.0%}) - the target is FLAT; reset the drawdown base to resume")
        elif dd <= -self.live.dd_half:
            g_t *= 0.5
            notes.append(f"drawdown {dd:.1%} from the high-water mark: gross halved")
        ecols = list(cut.etf_close.columns)
        sec_e = sector_etf_index(sig, ecols)
        port = Portfolio(len(sig.cols), p, ecols, sec_e, sig.sector_idx)
        port.etf_ok = np.isfinite(cut.etf_close.iloc[-1].to_numpy(float))
        port.load_dict(st.get("book", {}), sig.cols)
        x = DayInputs.from_signals(sig, i, p)
        T_w, Th, info = port.decide(x, g_t)

        deploy = self.live.capital_override if self.live.capital_override > 0 else equity * self.live.equity_fraction
        # what the engine owns: the broker is the truth for the symbols in its ledger (an unfilled order must not be remembered as a position)
        ledger = {k: int(v) for k, v in st.get("ledger", {}).items()}
        cur = dict(ledger)
        if self.tc is not None:
            try:
                broker = self.positions()
                for sym, q in ledger.items():
                    bq = int(broker.get(sym, 0))
                    if bq != q:
                        notes.append(f"ledger says {q:+d} {sym} but the broker holds {bq:+d} - planning from the broker's number")
                    cur[sym] = bq
            except Exception as e:                                                     # noqa: BLE001
                notes.append(f"could not read broker positions ({type(e).__name__}); planning from the ledger")
        stock_targets = {sig.cols[j]: float(T_w[j]) for j in np.where(T_w != 0)[0]}
        stocks_cur = {k: v for k, v in cur.items() if k not in D.FACTOR_ETFS}
        etf_cur = {k: v for k, v in cur.items() if k in D.FACTOR_ETFS}
        o_stock, tgt_map = self._orders(stock_targets, stocks_cur, cut, i, deploy, notes)
        # hedge from what will REALLY be held (whole shares, clipped, skipped names) - not from the unrounded target book
        px_row = cut.close.iloc[i]
        hedge_w: dict = {}
        if (p.w_eap > 0 and p.eap_hedge) or p.hedge == "etf":
            hedged_idx = set(np.where(port.eap_w != 0)[0]) if p.hedge != "etf" else set(range(len(sig.cols)))
            bsec = np.nan_to_num(x.b_sec, nan=1.0) if x.b_sec is not None else np.ones(len(sig.cols))
            sec_use = np.where(port.etf_ok[sec_e], sec_e, ecols.index("SPY") if "SPY" in ecols else sec_e)
            wr = np.zeros(len(sig.cols))
            col = {c: j for j, c in enumerate(sig.cols)}
            for sym, q in tgt_map.items():
                j = col.get(sym)
                if j is not None and j in hedged_idx and q > 0 and np.isfinite(px_row.get(sym, np.nan)) and deploy > 0:
                    wr[j] = q * float(px_row[sym]) / deploy
            hv = -np.bincount(sec_use, weights=wr * bsec, minlength=len(ecols))
            hedge_w = {ecols[j]: float(hv[j]) for j in np.where(np.abs(hv) > 1e-6)[0]}
        o_etf, tgt_etf = self._orders(hedge_w, etf_cur, cut, i, deploy, notes)
        orders = pd.concat([o_stock, o_etf], ignore_index=True)
        if len(orders) > self.live.max_orders:                                          # closing orders are never the ones dropped
            notes.append(f"{len(orders)} orders > max {self.live.max_orders}: all closing orders and the largest others are kept")
            ex = orders[orders.reason == "exit"]
            rest = orders[orders.reason != "exit"].sort_values("notional", ascending=False).head(max(self.live.max_orders - len(ex), 0))
            orders = pd.concat([ex, rest], ignore_index=True)
        ledger_after = dict(cur)
        for r in orders.itertuples():
            if r.ok:
                ledger_after[r.symbol] = int(r.tgt_qty)
        ledger_after = {k: v for k, v in ledger_after.items() if v != 0}
        targets = {**stock_targets, **hedge_w}
        expo = _exposures(T_w, x, sig)
        reg = {k: (float(v) if np.isfinite(v) else None) for k, v in sig.regime.iloc[i].items()}
        reg["shock_mult"] = mult
        reg["drawdown_from_hwm"] = float(dd)
        if info.get("n_eap") is not None:
            reg["n_eap"] = float(info["n_eap"])
        return Plan(decision_date=dec_date, deploy=float(deploy), equity=float(equity), g_target=g_t, shock_mult=mult, n_long=int((T_w > 0).sum()), n_short=int((T_w < 0).sum() + len(hedge_w)),
                    orders=orders, targets=targets, exposures=expo, regime=reg, notes=notes + ([info["note"]] if info.get("note") else []), book_after=port.to_dict(sig.cols),
                    ledger_after=ledger_after, hedge=hedge_w, vol_state={"var_u": st.get("var_u"), "n_u": int(st.get("n_u", 0))}, hwm=float(hwm))

    def _orders(self, targets: dict, cur: dict, panel, i: int, deploy: float, notes: list) -> tuple[pd.DataFrame, dict]:
        """Target weights -> whole-share orders against the shares the engine currently owns (``cur``). Returns (orders, {symbol: target shares})."""
        live = self.live
        px_all = pd.concat([panel.close.iloc[i], panel.etf_close.iloc[i]])
        px_all = px_all[~px_all.index.duplicated()]
        assets = None
        if os.path.exists(D.DATA_DIR + "/statarb_assets.parquet"):
            assets = pd.read_parquet(D.DATA_DIR + "/statarb_assets.parquet").set_index("symbol")
        rows, too_small, tgt_map = [], [], {}
        for sym in sorted(set(targets) | set(cur)):
            w = targets.get(sym, 0.0)
            px = float(px_all.get(sym, np.nan))
            have = int(cur.get(sym, 0))
            if not np.isfinite(px) or px <= 0:
                if have != 0 or w != 0:
                    notes.append(f"{sym}: no price on the decision session - left untouched (check whether it was delisted; use KILL or close it by hand)")
                tgt_map[sym] = have
                continue
            is_etf = sym in D.FACTOR_ETFS
            cap = live.max_name_pct * deploy
            tgt_notional = w * deploy if is_etf else max(-cap, min(cap, w * deploy))
            tgt = int(abs(tgt_notional) // px) * (1 if tgt_notional > 0 else -1)
            delta = tgt - have
            if tgt == 0 and have == 0 and w != 0:
                too_small.append(abs(w))
            if delta == 0:
                tgt_map[sym] = tgt
                continue
            closing = tgt == 0
            if abs(delta) * px < live.min_trade_usd and not closing:
                tgt_map[sym] = have
                continue
            if have != 0 and tgt != 0 and np.sign(have) != np.sign(tgt):
                delta, tgt = -have, 0                                                  # never flip in one step: close now, the book re-opens tomorrow if it still wants it
            ok, note = True, ""
            if tgt < 0 and live.require_etb and not is_etf:
                if assets is not None and sym in assets.index:
                    a = assets.loc[sym]
                    if not bool(a.get("shortable", False)) or not bool(a.get("etb", False)):
                        ok, note = False, "not shortable / not easy-to-borrow - skipped"
                else:
                    ok, note = False, "no asset record - cannot confirm it is shortable"
            kind = "exit" if closing else ("add" if have != 0 and np.sign(have) == np.sign(tgt) else "entry")
            tgt_map[sym] = tgt if ok else have
            rows.append(dict(symbol=sym, side="buy" if delta > 0 else "sell", qty=abs(delta), cur_qty=have, tgt_qty=tgt if ok else have, ref_px=px, notional=abs(delta) * px, target_w=w, ok=ok, note=note,
                             reason=kind))
        if too_small:
            notes.append(f"{len(too_small)} names skipped because one share costs more than the target position (they carry {sum(too_small) / max(sum(abs(v) for v in targets.values()), 1e-9):.0%} "
                         f"of the target gross) - the book is under-invested by that much; use more capital or accept it")
        df = pd.DataFrame(rows, columns=["symbol", "side", "qty", "cur_qty", "tgt_qty", "ref_px", "notional", "target_w", "ok", "note", "reason"])
        return df.reset_index(drop=True), tgt_map

    # ── execution ────────────────────────────────────────────────────────────────
    def submit(self, plan: Plan) -> list[dict]:
        """Send the plan. Raises unless the engine is ARMED. Updates the ledger / book state only for orders the broker accepted; a submit in which NOTHING was accepted leaves the session retryable."""
        if not self.live.armed:
            raise RuntimeError("DRY RUN: the engine is not armed, nothing is sent")
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        if self.tc is None:
            self.connect()
        if self.acct.get("blocked"):
            raise RuntimeError("trading is blocked on the account")
        if self.live.exec_mode == "opg":
            ok, msg = opg_window_open()
            if not ok:
                raise RuntimeError(msg + " - queue the orders later, or switch execution to 'day' for a test")
        results, st = [], self.load_state()
        if st.get("held_date") == str(plan.decision_date) and st.get("last_submit"):
            raise RuntimeError(f"orders for the session after {plan.decision_date:%Y-%m-%d} were already submitted ({st['last_submit'][:19]}); use Reconcile / Adopt, or plan again after the next close")
        if self.panel is not None:
            self.check_fresh(self.panel)
            if plan.decision_date != self.panel.dates[last_complete_session(self.panel)]:
                raise RuntimeError(f"stale plan: it is for {plan.decision_date:%Y-%m-%d} but the newest complete session is {self.panel.dates[last_complete_session(self.panel)]:%Y-%m-%d} - plan again")
        halted, why = self.loss_halt()
        if halted:
            self.log(f"DAILY-LOSS HALT: {why} - only closing orders are sent", "warn")
        ledger = {k: int(v) for k, v in st.get("ledger", {}).items()}
        tif = TimeInForce.OPG if self.live.exec_mode == "opg" else TimeInForce.DAY
        exits_first = plan.orders.assign(_k=plan.orders.reason.eq("exit").map({True: 0, False: 1})).sort_values("_k", kind="stable")
        accepted = 0
        for r in exits_first.itertuples():
            if not r.ok:
                results.append(dict(symbol=r.symbol, status="skipped", note=r.note))
                continue
            if halted and r.reason != "exit":
                results.append(dict(symbol=r.symbol, status="skipped", note="daily-loss halt: no new risk"))
                continue
            cid = f"{PREFIX}-{plan.decision_date:%Y%m%d}-{r.symbol}-{int(time.time()) % 100000}"
            try:
                o = self.tc.submit_order(MarketOrderRequest(symbol=r.symbol, qty=int(r.qty), side=OrderSide.BUY if r.side == "buy" else OrderSide.SELL, time_in_force=tif, client_order_id=cid[:120]))
                results.append(dict(symbol=r.symbol, status=str(getattr(o.status, "value", o.status)), id=str(o.id), qty=int(r.qty), side=r.side))
                ledger[r.symbol] = int(r.tgt_qty)
                accepted += 1
            except Exception as e:                                                     # noqa: BLE001
                results.append(dict(symbol=r.symbol, status="rejected", note=f"{type(e).__name__}: {str(e)[:140]}"))
            time.sleep(0.06)
        rejected = [r for r in results if r.get("status") == "rejected"]
        if rejected:
            hedge_rej = [r["symbol"] for r in rejected if r["symbol"] in D.FACTOR_ETFS]
            self.log(f"{len(rejected)} order(s) REJECTED" + (f" - including hedge ETFs {hedge_rej}: the book may be unhedged, reconcile now" if hedge_rej else " - reconcile and re-plan"), "err")
        if accepted == 0:
            self._log_orders(plan, results)
            return results                                                              # nothing went out: leave the state alone so the session can be retried
        ledger = {k: v for k, v in ledger.items() if v}
        st["ledger"] = ledger
        st["book"] = _book_with_ledger(plan.book_after, ledger)
        st["held_w"] = {sym: float(w) for sym, w in plan.targets.items() if ledger.get(sym)}
        st["held_date"] = str(plan.decision_date)
        st["last_submit"] = str(pd.Timestamp.now())
        st["var_u"], st["n_u"] = plan.vol_state.get("var_u"), int(plan.vol_state.get("n_u", 0))
        st["hwm"] = plan.hwm or st.get("hwm")
        self.save_state(st)
        self._log_orders(plan, results)
        return results

    def _log_orders(self, plan: Plan, results: list) -> None:
        path = os.path.join(D.REPORTS, f"statarb_orders_{'paper' if self.live.paper else 'live'}.csv")
        df = pd.DataFrame(results)
        df.insert(0, "decision_date", str(plan.decision_date.date()))
        df.insert(0, "logged", str(pd.Timestamp.now()))
        df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

    def reset_drawdown_base(self) -> float:
        """Set the high-water mark to today's equity (after a drawdown stop has been reviewed)."""
        st = self.load_state()
        st["hwm"] = float(self.acct.get("equity", 0.0)) if self.acct else None
        self.save_state(st)
        return float(st["hwm"] or 0.0)

    def reconcile(self) -> pd.DataFrame:
        st = self.load_state()
        ledger = {k: int(v) for k, v in st.get("ledger", {}).items()}
        broker = self.positions()
        rows = []
        for s in sorted(set(ledger)):
            rows.append(dict(symbol=s, ledger=ledger.get(s, 0), broker=int(broker.get(s, 0)), diff=int(broker.get(s, 0)) - ledger.get(s, 0)))
        return pd.DataFrame(rows, columns=["symbol", "ledger", "broker", "diff"])

    def adopt_broker(self) -> int:
        """Make the ledger match the broker for the symbols the engine owns (after a partial fill, reject or manual change). Book states of names with no position are dropped."""
        st = self.load_state()
        broker = self.positions()
        new = {s: int(broker[s]) for s in st.get("ledger", {}) if s in broker and int(broker[s]) != 0}
        n = sum(1 for s in st.get("ledger", {}) if new.get(s, 0) != st["ledger"][s])
        st["ledger"] = new
        bk = st.get("book", {})
        if bk:
            bk.setdefault("rev", {})["positions"] = {sym: v for sym, v in (bk.get("rev", {}).get("positions") or {}).items() if sym in new}
            bk.setdefault("eap", {})["w"] = {sym: w for sym, w in (bk.get("eap", {}).get("w") or {}).items() if sym in new}
            st["book"] = bk
        self.save_state(st)
        return n

    def cancel_open(self) -> tuple[int, list]:
        n, errs = 0, []
        try:
            orders = self.tc.get_orders()
        except Exception as e:                                                         # noqa: BLE001
            return 0, [f"could not list open orders: {type(e).__name__}: {str(e)[:100]}"]
        for o in orders:
            if str(o.client_order_id).startswith(PREFIX + "-"):
                try:
                    self.tc.cancel_order_by_id(o.id)
                    n += 1
                except Exception as e:                                                 # noqa: BLE001
                    errs.append(f"{o.symbol}: cancel failed ({type(e).__name__}: {str(e)[:80]})")
        return n, errs

    def flatten(self) -> list[dict]:
        """KILL: cancel the engine's open orders, then close what the ENGINE owns - min(ledger, broker) per symbol, same direction only, so shares held by other strategies in the account are never
        touched - with market orders (day orders if the market is open, else queued for the open). The ledger forgets a symbol only when its closing order was accepted. Works in dry run too:
        it is an emergency action behind a typed confirmation."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        if self.tc is None:
            self.connect()
        out: list = []
        n, errs = self.cancel_open()
        for e in errs:
            out.append(dict(symbol="*", qty=0, status=e))
        st = self.load_state()
        broker = self.positions()
        is_open = bool(self.tc.get_clock().is_open)
        ledger = {k: int(v) for k, v in st.get("ledger", {}).items()}
        for sym, lq in list(ledger.items()):
            bq = int(broker.get(sym, 0))
            if bq == 0 or np.sign(bq) != np.sign(lq):
                if bq == 0:
                    ledger.pop(sym, None)
                else:
                    out.append(dict(symbol=sym, qty=bq, status="skipped: broker position is on the other side of the ledger - close it by hand"))
                continue
            q = min(abs(lq), abs(bq)) * (1 if bq > 0 else -1)
            try:
                self.tc.submit_order(MarketOrderRequest(symbol=sym, qty=abs(q), side=OrderSide.SELL if q > 0 else OrderSide.BUY, time_in_force=TimeInForce.DAY if is_open else TimeInForce.OPG,
                                                        client_order_id=f"{PREFIX}-KILL-{sym}-{int(time.time()) % 100000}"))
                out.append(dict(symbol=sym, qty=q, status="sent"))
                ledger.pop(sym, None)
            except Exception as e:                                                     # noqa: BLE001
                out.append(dict(symbol=sym, qty=q, status=f"FAILED {type(e).__name__}: {str(e)[:100]} - still in the ledger"))
        st["ledger"] = ledger
        keep = set(ledger)
        bk = st.get("book", {})
        if bk:
            bk.setdefault("rev", {})["positions"] = {k: v for k, v in (bk.get("rev", {}).get("positions") or {}).items() if k in keep}
            bk.setdefault("eap", {})["w"] = {k: v for k, v in (bk.get("eap", {}).get("w") or {}).items() if k in keep}
            st["book"] = bk
        st["held_w"] = {k: v for k, v in (st.get("held_w") or {}).items() if k in keep}
        self.save_state(st)
        return out


# ══ helpers ═════════════════════════════════════════════════════════════════════════
def _truncate(panel: D.Panel, i: int) -> D.Panel:
    """A copy of the panel that ends at row ``i`` (so signals never see a partial or later session)."""
    q = object.__new__(D.Panel)
    q.__dict__.update(panel.__dict__)
    for k in ("open_all", "close_all", "volume_all", "open", "close", "volume", "etf_close", "etf_open"):
        q.__dict__[k] = getattr(panel, k).iloc[:i + 1]
    q.dates = panel.dates[:i + 1]
    return q


def _exposures(T_w: np.ndarray, x: DayInputs, sig) -> dict:
    """Net exposure of the target book to each risk factor (weights are fractions of equity)."""
    g = float(np.abs(T_w).sum()) or 1.0
    f = lambda a: float(np.nansum(T_w * np.nan_to_num(a)) / g)
    out = {"gross": g, "net_dollar": float(T_w.sum() / g), "beta_spy": f(x.b_spy), "oil_beta": f(x.b_oil), "rates_beta": f(x.b_tlt), "usd_beta": f(x.b_usd),
           "momentum": f(x.mom_z), "volatility": f(x.vol_z), "size": f(x.size_z)}
    for k in range(min(3, x.L.shape[1])):
        out[f"pca{k + 1}"] = f(x.L[:, k])
    for k, name in enumerate(sig.sector_names):
        m = x.sector_idx == k
        out[f"sector:{name}"] = float(T_w[m].sum() / g)
    return out


def _book_with_ledger(book_after: dict, ledger: dict) -> dict:
    """Keep only the book states whose orders the broker accepted (a rejected entry must not be remembered as a position)."""
    rev = book_after.get("rev", {})
    keep_pos = {sym: v for sym, v in (rev.get("positions") or {}).items() if sym in ledger}
    eap = book_after.get("eap", {})
    keep_eap = {sym: w for sym, w in (eap.get("w") or {}).items() if sym in ledger}
    return {"rev": {"positions": keep_pos, "g_applied": rev.get("g_applied")}, "eap": {"w": keep_eap, "g_applied": eap.get("g_applied")}}
