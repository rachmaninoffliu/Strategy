"""
Notifications that reach you AWAY from the PC (the watchdog's pop-up window is no use when you are not at the desk).

Set the environment variable LEVPAIRS_NOTIFY_URL to a URL that accepts an HTTP POST whose body is the message text - the ntfy convention (https://ntfy.sh/<your-secret-topic> with the ntfy phone app subscribed to that
topic, or a self-hosted ntfy server; anything that takes a POST body works). Nothing else is configured, nothing is installed, and with the variable unset this module does nothing.
What is sent is the alert text itself ("ENGINE DOWN ... the account holds {'TQQQ': 100.0, ...}", the evening summary): pick a topic nobody can guess, and remember that the message passes through that service.

It never raises, never blocks its caller for more than a moment (the POST runs in a daemon thread), and the same message (digits ignored) goes out at most once in 10 minutes.
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Callable, Optional

URL_ENV = "LEVPAIRS_NOTIFY_URL"
_last: dict = {}


def configured() -> bool:
    return bool(os.environ.get(URL_ENV))


def send(msg: str, title: str = "Lev-ETF pairs", post: Optional[Callable] = None, wait: bool = False, dedupe_s: float = 600.0) -> bool:
    """POST ``msg`` to the configured URL. Returns True if a notification was sent (or started), False if nothing is configured or the message was a duplicate."""
    url = os.environ.get(URL_ENV)
    if not url:
        return False
    key = re.sub(r"\d+", "#", msg)[:80]
    now = time.time()
    if now - _last.get(key, 0.0) < dedupe_s:
        return False
    _last[key] = now

    def go() -> None:
        try:
            if post is None:
                import requests
                requests.post(url, data=msg[:1500].encode("utf-8"), headers={"Title": title, "Priority": "high"}, timeout=(5, 10))
            else:
                post(url, data=msg[:1500].encode("utf-8"), headers={"Title": title, "Priority": "high"}, timeout=(5, 10))
        except Exception:                                                                # noqa: BLE001 - a notification must never break the thing it reports on
            pass
    if wait:
        go()
    else:
        threading.Thread(target=go, daemon=True).start()
    return True
