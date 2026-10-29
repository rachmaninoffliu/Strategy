"""
Alpaca API keys for every app in this PyCharm project (one identical copy per project; keep them identical).

Order:
  1. the key file in the project folder - "Alpaca Key.txt" / "Alpaca Kay.txt" / "Alpaca Keys.txt", found by walking up from this file - with the lines
         Key ID: PK...
         Secret Key: ...
     (ALPACA_KEY_FILE=<path> points to another file; ALPACA_KEY_FILE=none switches the file off - the tests do, so they never see real keys).
     A paper key (PK...) is used for paper connections only, a live key (AK...) for live connections only.
  2. the environment: ALPACA_API_KEY / ALPACA_SECRET_KEY
  3. ~/.claude.json -> mcpServers -> alpaca-paper / alpaca-live -> env
The key file is git-ignored: never copy keys into code, logs or reports.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

KEY_FILE_NAMES = ("Alpaca Key.txt", "Alpaca Kay.txt", "Alpaca Keys.txt", "Alpaca Key", "Alpaca Kay")


def find_key_file(start: Optional[str] = None) -> Optional[str]:
    env = os.environ.get("ALPACA_KEY_FILE")
    if env is not None:
        return None if env.strip().lower() in ("", "none", "off", "-") else (env if os.path.isfile(env) else None)
    d = os.path.abspath(start or os.path.dirname(os.path.abspath(__file__)))
    while True:
        for n in KEY_FILE_NAMES:
            p = os.path.join(d, n)
            if os.path.isfile(p):
                return p
        up = os.path.dirname(d)
        if up == d:
            return None
        d = up


def read_key_file(path: str) -> tuple[str, str]:
    txt = open(path, encoding="utf-8-sig").read()
    k = re.search(r"(?:API\s*)?Key\s*ID\s*[:=]\s*(\S+)", txt, re.I)
    s = re.search(r"Secret(?:\s*Key)?\s*[:=]\s*(\S+)", txt, re.I)
    if not k or not s:
        raise RuntimeError(f"{os.path.basename(path)}: expected a 'Key ID: ...' line and a 'Secret Key: ...' line")
    return k.group(1).strip(), s.group(1).strip()


def _file_keys(paper: bool) -> Optional[tuple[str, str]]:
    p = find_key_file()
    if not p:
        return None
    k, s = read_key_file(p)
    if k.upper().startswith("PK") and not paper:
        return None                                                                          # a paper key never serves a live (real-money) connection
    if k.upper().startswith("AK") and paper:
        return None
    return k, s


def source(paper: bool = True) -> str:
    """Where creds(paper) takes the keys from: 'file', 'env', 'claude.json' or 'none'."""
    if _file_keys(paper):
        return "file"
    if paper and os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"):
        return "env"
    try:
        env = json.load(open(os.path.expanduser("~/.claude.json"), encoding="utf-8")).get("mcpServers", {}).get("alpaca-paper" if paper else "alpaca-live", {}).get("env", {})
        if env.get("ALPACA_API_KEY") and not str(env.get("ALPACA_API_KEY")).startswith("YOUR_"):
            return "claude.json"
    except (OSError, ValueError):
        pass
    return "none"


def creds(paper: bool = True) -> tuple[str, str]:
    fk = _file_keys(paper)
    if fk:
        return fk
    k, s = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if paper and k and s:
        return k, s
    node = "alpaca-paper" if paper else "alpaca-live"
    try:
        cfg = json.load(open(os.path.expanduser("~/.claude.json"), encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise RuntimeError(f"no Alpaca keys: put 'Alpaca Key.txt' (Key ID / Secret Key) in the project folder ({e})") from e
    env = cfg.get("mcpServers", {}).get(node, {}).get("env", {})
    k, s = env.get("ALPACA_API_KEY"), env.get("ALPACA_SECRET_KEY")
    if not k or not s or str(k).startswith("YOUR_"):
        raise RuntimeError(f"no Alpaca {'paper' if paper else 'live'} keys: put 'Alpaca Key.txt' (Key ID / Secret Key) in the project folder")
    return k, s


def describe(paper: bool = True) -> str:
    """Where the keys come from, for a status line (never the keys)."""
    src = source(paper)
    if src == "file":
        return f"key file {os.path.basename(find_key_file() or '')}"
    return {"env": "environment variables", "claude.json": "~/.claude.json", "none": "no keys"}[src]
