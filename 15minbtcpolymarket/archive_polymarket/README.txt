Retired on 2026-10-12: the app now signals AND trades on Kalshi only.
These files belong to the earlier Polymarket US setup (client/broker, cross-venue logger and scans, their data). They are kept here only for reference and are not used by the app.
The Polymarket US public book is a cached snapshot (up to ~30 s old, strict rate limit) and has only ~2 weeks of history, so it cannot drive or validate this strategy.

live_history/session_20261011_181945_live.csv: the 4 real LIVE trades made on Polymarket US on 2026-10-11 (all losses, -$50.52 in total).
Moved here on 2026-10-14 so the app's trade history only shows Kalshi trades; the app also ignores any trade whose venue is "polymarket" (trader/ledger.py RETIRED_VENUES).
