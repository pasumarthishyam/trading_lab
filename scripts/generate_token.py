"""
Zerodha Access Token Generator
===============================

Interactive CLI script for daily Zerodha KiteConnect login.

Flow:
    1. Loads ``ZERODHA_API_KEY`` and ``ZERODHA_API_SECRET`` from ``.env``
    2. Prints the Kite login URL
    3. User logs in via browser, pastes the redirect URL back
    4. Extracts ``request_token`` from the redirect URL
    5. Exchanges it for an ``access_token`` via ``generate_session()``
    6. Writes the access token back into ``.env``
    7. Verifies by fetching the user profile

Usage:
    python scripts/generate_token.py

Notes:
    - Run this once per trading day (tokens expire at 6 AM IST next day)
    - ``ZERODHA_API_KEY`` and ``ZERODHA_API_SECRET`` must be set in ``.env``
      before first run
    - This is a user-facing CLI tool, so it uses print()/input() for
      interaction — the no-print rule applies only to library code
"""

import logging
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Project root: scripts/ is one level below trading_lab/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
ENV_PATH: Path = PROJECT_ROOT / ".env"


def _load_credentials() -> tuple[str, str]:
    """Load API key and secret from .env.

    Returns
    -------
    tuple[str, str]
        (api_key, api_secret)

    Raises
    ------
    SystemExit
        If either credential is missing or still a placeholder.
    """
    import os

    load_dotenv(ENV_PATH)

    api_key = os.environ.get("ZERODHA_API_KEY", "")
    api_secret = os.environ.get("ZERODHA_API_SECRET", "")

    if not api_key or api_key == "your_api_key_here":
        print("\n[ERROR]  ZERODHA_API_KEY is not set in .env")
        print(f"   Edit: {ENV_PATH}")
        raise SystemExit(1)

    if not api_secret or api_secret == "your_api_secret_here":
        print("\n[ERROR]  ZERODHA_API_SECRET is not set in .env")
        print(f"   Edit: {ENV_PATH}")
        raise SystemExit(1)

    return api_key, api_secret


def _extract_request_token(redirect_url: str) -> str:
    """Extract request_token from the redirect URL.

    Parameters
    ----------
    redirect_url : str
        The full URL copied from the browser after Kite login.
        Example: ``https://127.0.0.1/?request_token=abc123&action=login&...``

    Returns
    -------
    str
        The extracted request_token.

    Raises
    ------
    SystemExit
        If no request_token found in the URL.
    """
    parsed = urlparse(redirect_url.strip())
    params = parse_qs(parsed.query)

    if "request_token" in params:
        return params["request_token"][0]

    # Fallback: try regex in case URL format is unusual.
    match = re.search(r"request_token=([a-zA-Z0-9]+)", redirect_url)
    if match:
        return match.group(1)

    print("\n[ERROR]  Could not find request_token in the URL you pasted.")
    print("   Make sure you copied the full redirect URL from the browser.")
    raise SystemExit(1)


def _update_env_token(access_token: str) -> None:
    """Write access_token into .env by replacing the existing line.

    Parameters
    ----------
    access_token : str
        The new access token to write.
    """
    if not ENV_PATH.exists():
        print(f"\n[ERROR]  .env file not found at {ENV_PATH}")
        raise SystemExit(1)

    lines = ENV_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    updated = False

    for i, line in enumerate(lines):
        if line.startswith("ZERODHA_ACCESS_TOKEN="):
            lines[i] = f"ZERODHA_ACCESS_TOKEN={access_token}\n"
            updated = True
            break

    if not updated:
        # Key doesn't exist yet — append it.
        lines.append(f"ZERODHA_ACCESS_TOKEN={access_token}\n")

    ENV_PATH.write_text("".join(lines), encoding="utf-8")
    logger.info("Access token written to %s", ENV_PATH)


def main() -> None:
    """Run the interactive token generation flow."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("  Zerodha KiteConnect — Daily Token Generator")
    print("=" * 60)

    # Step 1: Load credentials.
    api_key, api_secret = _load_credentials()
    print(f"\n[OK]  API Key loaded: {api_key[:4]}...{api_key[-4:]}")

    # Step 2: Import KiteConnect and build login URL.
    from kiteconnect import KiteConnect  # type: ignore[import-untyped]

    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()


    print(f"\n>>> Open this URL in your browser and log in:\n")
    print(f"    {login_url}\n")

    # Step 3: Get redirect URL from user.
    redirect_url = input(">>> Paste the redirect URL here: ")

    # Step 4: Extract request_token.
    request_token = _extract_request_token(redirect_url)
    print(f"\n[OK]  Request token extracted: {request_token[:8]}...")

    # Step 5: Exchange for access token.
    print("\n...   Exchanging for access token...")
    session_data = kite.generate_session(
        request_token=request_token,
        api_secret=api_secret,
    )
    access_token: str = session_data["access_token"]
    print(f"[OK]  Access token received: {access_token[:8]}...")

    # Step 6: Write to .env.
    _update_env_token(access_token)
    print(f"[OK]  Token written to {ENV_PATH}")

    # Step 7: Verify by fetching profile.
    kite.set_access_token(access_token)
    profile = kite.profile()
    user_id = profile.get("user_id", "unknown")
    user_name = profile.get("user_name", "unknown")

    print(f"\n[SUCCESS]  Login successful!")
    print(f"   User: {user_name} ({user_id})")
    print(f"   Token valid until ~6:00 AM IST tomorrow")
    print(f"\n   You can now run: python scripts/fetch_initial_data.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
