"""One-time E*TRADE OAuth 1.0a handshake.

Run this once on first setup, and again whenever the access token has expired
at end-of-day (E*TRADE kills access tokens at midnight US/Eastern):

    python auth_handshake.py

Steps performed:
  1. Fetch a temporary request token from /oauth/request_token (HMAC-SHA1 signed)
  2. Open the authorize URL in your default browser
  3. Prompt for the 5-character verification PIN E*TRADE displays
  4. Exchange the PIN for an access token + secret
  5. Persist tokens to TOKEN_STORE_PATH so the FastAPI server can run headless
"""
from __future__ import annotations

import sys
import time
import webbrowser

from requests_oauthlib import OAuth1Session

from app.config import get_settings
from app.token_store import TokenBundle, TokenStore


REQUEST_TOKEN_URL = "https://api.etrade.com/oauth/request_token"
ACCESS_TOKEN_URL = "https://api.etrade.com/oauth/access_token"
AUTHORIZE_URL_TPL = "https://us.etrade.com/e/t/etws/authorize?key={key}&token={token}"


def main() -> int:
    settings = get_settings()
    store = TokenStore(settings.token_store_file)

    print(f"[*] E*TRADE env : {settings.etrade_env}")
    print(f"[*] token store : {settings.token_store_file}")

    print("[1/3] Requesting temporary request token ...")
    req_sess = OAuth1Session(
        client_key=settings.etrade_consumer_key,
        client_secret=settings.etrade_consumer_secret,
        callback_uri="oob",
        signature_method="HMAC-SHA1",
    )
    try:
        rt = req_sess.fetch_request_token(REQUEST_TOKEN_URL)
    except Exception as exc:
        print(f"FAILED: request_token error: {exc}", file=sys.stderr)
        return 1

    req_token = rt["oauth_token"]
    req_secret = rt["oauth_token_secret"]

    authorize_url = AUTHORIZE_URL_TPL.format(
        key=settings.etrade_consumer_key, token=req_token
    )
    print("[2/3] Open this URL, log in, click Accept, then copy the PIN:")
    print(f"  {authorize_url}")
    try:
        webbrowser.open(authorize_url, new=2)
    except Exception:
        pass

    pin = input("Paste the verification PIN E*TRADE displays: ").strip()
    if not pin:
        print("FAILED: empty PIN", file=sys.stderr)
        return 1

    print("[3/3] Exchanging PIN for access token ...")
    exchange = OAuth1Session(
        client_key=settings.etrade_consumer_key,
        client_secret=settings.etrade_consumer_secret,
        resource_owner_key=req_token,
        resource_owner_secret=req_secret,
        verifier=pin,
        signature_method="HMAC-SHA1",
    )
    try:
        at = exchange.fetch_access_token(ACCESS_TOKEN_URL)
    except Exception as exc:
        print(f"FAILED: access_token error: {exc}", file=sys.stderr)
        return 1

    now = time.time()
    store.save(
        TokenBundle(
            oauth_token=at["oauth_token"],
            oauth_token_secret=at["oauth_token_secret"],
            environment=settings.etrade_env,
            created_at=now,
            last_renewed_at=now,
        )
    )
    print(f"[OK] tokens persisted to {settings.token_store_file}")
    print("Start the bridge with:  uvicorn app.main:app --host 127.0.0.1 --port 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
