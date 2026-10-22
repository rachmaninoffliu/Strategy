"""Global constants, hurdle rates, API URLs and runtime settings for prediction_arb."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

APP_NAME = "Prediction Arb  |  Kalshi x Polymarket"
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "arb_data.db"

# ── Endpoints (public, read-only, no keys) ────────────────────────────────────
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_WEB = "https://kalshi.com/events"             # + /{EVENT_TICKER}  (verified: resolves to that exact event)
POLY_GAMMA = "https://gamma-api.polymarket.com"
POLY_US_GATEWAY = "https://gateway.polymarket.us"        # public, no key: Polymarket US (CFTC-regulated, US-tradable)
POLY_US_WEB = "https://polymarket.us"
POLY_CLOB = "https://clob.polymarket.com"
POLY_WEB = "https://polymarket.com/event"

# ── Hurdle-rate maths ─────────────────────────────────────────────────────────
HURDLE_APY = 0.07          # 7.0 %
MIN_DAYS = 0.25            # D = max(days_to_settle, 0.25)
DAYS_PER_YEAR = 365.25
APY_EXP_CAP = 60.0         # cap on the exponent so APY never overflows a float
APY_GREEN = 0.15           # UI colour thresholds (APY > 15 % green, 7-15 % yellow)

# ── HTTP behaviour ────────────────────────────────────────────────────────────
USER_AGENT = "prediction-arb/1.0 (read-only research client)"
HTTP_TIMEOUT_S = 45
MAX_RETRIES = 5
KALSHI_RPS = 8.0           # Kalshi public read limit is well above this
POLY_GAMMA_RPS = 4.0
POLY_US_RPS = 8.0
POLY_CLOB_RPS = 8.0
CONCURRENCY = 8

# ── Settlement-lag assumptions (capital stays locked past the scheduled time) ─
KALSHI_SETTLE_LAG_H = 2.0
POLY_SETTLE_LAG_H = 4.0    # UMA liveness is ~2h if undisputed; disputes take days


@dataclass
class FeeConfig:
    """Fee model. Kalshi taker fee = ceil_to_cent(coef * C * P * (1-P)).

    Polymarket charges a taker fee only on markets that have ``feesEnabled``;
    the per-market ``feeSchedule`` (rate, exponent) is used in ``schedule`` mode:
    fee = C * rate * (P * (1-P)) ** exponent.
    """
    kalshi_coef: float = 0.07
    poly_mode: str = "schedule"      # "schedule" | "flat" | "zero"
    poly_flat_bps: float = 0.0       # used when poly_mode == "flat" (bps of notional)
    extra_bps: float = 0.0           # additional bps of notional on every leg


@dataclass
class LongShotConfig:
    """Collection bounds for the long-shot screen (the UI tab filters tighter on the collected set)."""
    max_prob: float = 0.10             # market-implied P(YES) ceiling at collection time
    min_prob: float = 0.001
    min_volume_usd: float = 10_000.0   # cash volume floor ...
    min_open_interest: float = 25_000.0  # ... OR open-interest notional floor (Kalshi reports OI; Polymarket does not)
    risk_free: float = 0.05            # T-bill yield the tail-seller carry is compared with


@dataclass
class ScanSettings:
    """Live-scanner settings (editable from the Live tab)."""
    hurdle_apy: float = HURDLE_APY
    min_days: float = MIN_DAYS
    min_depth_usd: float = 25.0
    min_confidence: float = 0.80
    min_volume: float = 250.0
    kalshi_max_pages: int = 60
    poly_max_pages: int = 60
    use_l2_books: bool = True
    max_book_candidates: int = 40
    max_basket_legs: int = 40
    max_net_return: float = 0.30      # sanity cap: bigger 'edges' are almost always definition mismatches
    mece_mid_lo: float = 0.88         # exhaustive-set sanity: sum of YES mid-prices must be ~1
    mece_mid_hi: float = 1.12
    date_tol_hours: float = 24.0
    subset_tol_hours: float = 2.0
    include_types: tuple = ("parity", "mece", "subset")
    longshot: LongShotConfig = None   # filled in __post_init__
    longshots_only: bool = False      # fast path: skip matching / arbitrage detection
    poly_source: str = "us"           # "us" = Polymarket US (tradable by US persons) | "global" = polymarket.com (NOT)
    us_min_volume_usd: float = 1_000.0  # Polymarket US: server-side market volume floor for the arbitrage universe
    us_only: bool = True              # drop anything a US person cannot trade
    fees: FeeConfig = None            # filled in __post_init__
    kalshi_lag_h: float = KALSHI_SETTLE_LAG_H
    poly_lag_h: float = POLY_SETTLE_LAG_H
    refresh_seconds: int = 60

    def __post_init__(self):
        if self.fees is None:
            self.fees = FeeConfig()
        if self.longshot is None:
            self.longshot = LongShotConfig()
