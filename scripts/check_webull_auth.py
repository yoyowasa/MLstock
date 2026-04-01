from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv


def _mask(value: str | None) -> str:
    if not value:
        return "MISSING"
    text = str(value)
    if len(text) <= 8:
        return f"SET(len={len(text)})"
    return f"{text[:4]}...{text[-4:]}(len={len(text)})"


def _build_clients():
    from webullsdkcore.client import ApiClient
    from webullsdkcore.common.region import Region
    from webullsdktrade.api import API

    base_url = os.getenv("WEBULL_BASE_URL", "https://api.webull.co.jp").strip()
    region = os.getenv("WEBULL_REGION", "jp").strip().lower() or "jp"
    app_key = os.getenv("WEBULL_APP_KEY", "").strip()
    app_secret = os.getenv("WEBULL_APP_SECRET", "").strip()
    access_token = os.getenv("WEBULL_ACCESS_TOKEN", "").strip()
    token_dir = os.getenv("WEBULL_OPENAPI_TOKEN_DIR", "").strip()

    region_value = Region.JP.value if region == "jp" else region
    api_client = ApiClient(
        app_key,
        app_secret,
        region_value,
        verify=True,
        timeout=30,
        connect_timeout=10,
    )
    api_client._stream_logger_set = True
    api_client._file_logger_set = True
    api_client.add_endpoint(region, base_url.replace("https://", "").replace("http://", ""))

    return {
        "api_client": api_client,
        "api": API(api_client),
        "base_url": base_url,
        "region": region,
        "access_token": access_token,
        "token_dir": token_dir,
        "app_key": app_key,
        "app_secret": app_secret,
    }


def _print_json(title: str, payload) -> None:
    print(f"\n[{title}]")
    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Webull auth and account access")
    parser.add_argument("--dotenv", type=Path, default=Path(".env"), help="Path to .env file")
    parser.add_argument("--token", type=str, default=None, help="Temporary token string for create/check")
    args = parser.parse_args()

    load_dotenv(args.dotenv, override=False)
    clients = _build_clients()

    print("[env]")
    print(f"WEBULL_BASE_URL={clients['base_url']}")
    print(f"WEBULL_REGION={clients['region']}")
    print(f"WEBULL_APP_KEY={_mask(clients['app_key'])}")
    print(f"WEBULL_APP_SECRET={_mask(clients['app_secret'])}")
    print(f"WEBULL_ACCESS_TOKEN={_mask(clients['access_token'])}")
    print(f"WEBULL_OPENAPI_TOKEN_DIR={clients['token_dir'] or 'MISSING'}")

    api = clients["api"]

    try:
        response = api.account.get_app_subscriptions()
        _print_json("subscriptions_status", getattr(response, "status_code", None))
        _print_json("subscriptions_body", response.json())
    except Exception as exc:
        _print_json("subscriptions_error", repr(exc))

    try:
        subscriptions = api.account.get_app_subscriptions().json()
        if subscriptions:
            account_id = subscriptions[0].get("account_id")
            balance = api.account.get_account_balance(account_id, "USD")
            _print_json("balance_status", getattr(balance, "status_code", None))
            _print_json("balance_body", balance.json())
            positions = api.account.get_account_position(account_id, page_size=20)
            _print_json("positions_status", getattr(positions, "status_code", None))
            _print_json("positions_body", positions.json())
    except Exception as exc:
        _print_json("account_detail_error", repr(exc))

    token_input = args.token or clients["access_token"]
    if token_input:
        _print_json("token_note", "JP PDF flow does not require access token for the tested account APIs.")
    else:
        _print_json("token_check", "SKIPPED(no token provided)")


if __name__ == "__main__":
    main()
