from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    etrade_consumer_key: str
    etrade_consumer_secret: str
    etrade_account_id_key: str
    webhook_secret_passphrase: str

    etrade_env: str = "sandbox"
    token_store_path: str = "~/.etrade_bridge/tokens.json"
    renewal_interval_minutes: int = 90

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    @property
    def api_base(self) -> str:
        if self.etrade_env.lower() == "production":
            return "https://api.etrade.com"
        return "https://apisb.etrade.com"

    @property
    def oauth_base(self) -> str:
        # OAuth request/access/renew endpoints are the same host for sandbox + prod.
        return "https://api.etrade.com"

    @property
    def token_store_file(self) -> Path:
        return Path(self.token_store_path).expanduser().resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
