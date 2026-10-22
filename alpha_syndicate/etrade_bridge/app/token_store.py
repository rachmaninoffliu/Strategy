from __future__ import annotations

import json
import os
import stat
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Optional


@dataclass
class TokenBundle:
    oauth_token: str
    oauth_token_secret: str
    environment: str
    created_at: float
    last_renewed_at: float


class TokenStore:
    """File-backed OAuth token cache. Thread-safe within a single process."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = Lock()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> Optional[TokenBundle]:
        with self._lock:
            if not self._path.exists():
                return None
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                return TokenBundle(**raw)
            except (json.JSONDecodeError, TypeError, KeyError):
                return None

    def save(self, tokens: TokenBundle) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(json.dumps(asdict(tokens), indent=2), encoding="utf-8")
            os.replace(tmp, self._path)
            try:
                # POSIX: owner read/write only. No-op on Windows.
                os.chmod(self._path, stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass

    def touch_renewed(self) -> None:
        bundle = self.load()
        if bundle is None:
            return
        bundle.last_renewed_at = time.time()
        self.save(bundle)
