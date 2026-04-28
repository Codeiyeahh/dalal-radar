"""
Market Agent Module.

Fetches Bulk and Block deals from NSE for the last 24 hours.

Primary strategy : nselib.capital_market  (bulk_deal_data / block_deals_data)
Fallback strategy: jugaad-data NSELive + direct NSE httpx when nselib fails.

Exposed functions
-----------------
get_all_deals()              -> list[dict]
get_deals_by_symbol(symbol)  -> list[dict]
format_deals_for_rag(deals)  -> str
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap path so we can import config when run as __main__
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import logger  # noqa: E402

# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
_RESPONSE_DATE_FMTS = ["%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"]


def _parse_date(date_str: str) -> Optional[datetime]:
    """Try multiple common NSE date formats and return a datetime or None."""
    cleaned = str(date_str).strip().upper()
    for fmt in _RESPONSE_DATE_FMTS:
        try:
            return datetime.strptime(cleaned, fmt.upper())
        except ValueError:
            continue
    return None


def _is_within_24h(date_str: str) -> bool:
    """Return True if *date_str* falls within the last 24 hours."""
    cutoff = datetime.now() - timedelta(hours=24)
    dt = _parse_date(date_str)
    # If we can't parse it, include the record (NSE may return today-only data)
    if dt is None:
        return True
    return dt >= cutoff


# ---------------------------------------------------------------------------
# Normalise a raw DataFrame row (dict) into the project's standard schema
# ---------------------------------------------------------------------------
# nselib column names (verified against live data):
#   Date, Symbol, SecurityName, ClientName, Buy/Sell,
#   QuantityTraded, TradePrice/Wght.Avg.Price, Remarks

def _normalise(raw: dict, deal_type: str) -> dict:
    """Convert a raw NSE deal record into a clean, uniform dict."""

    # Quantity comes as a comma-separated string like '13,00,000'
    raw_qty = str(raw.get("QuantityTraded", raw.get("quantity", "N/A")))
    qty = raw_qty.replace(",", "").strip()

    raw_price = str(raw.get("TradePrice/Wght.Avg.Price", raw.get("price", "N/A")))
    price = raw_price.replace(",", "").strip()

    return {
        "symbol":      str(raw.get("Symbol", raw.get("symbol", "N/A"))).strip().upper(),
        "deal_type":   deal_type,
        "client_name": str(raw.get("ClientName", raw.get("clientName", "N/A"))).strip(),
        "quantity":    qty,
        "price":       price,
        "deal_date":   str(raw.get("Date", raw.get("date", "N/A"))).strip(),
    }


# ---------------------------------------------------------------------------
# Strategy 1 — nselib (preferred, handles NSE cookies internally)
# ---------------------------------------------------------------------------

def _fetch_deals_nselib() -> tuple[list[dict], list[dict]]:
    """
    Use ``nselib.capital_market`` to pull bulk & block deal DataFrames.

    Returns (bulk_records, block_records) as lists of dicts.
    """
    from nselib import capital_market  # lazy import so failure is catchable

    raw_bulk: pd.DataFrame = capital_market.bulk_deal_data(period="1D")
    raw_block: pd.DataFrame = capital_market.block_deals_data(period="1D")

    bulk_recs = raw_bulk.to_dict(orient="records") if not raw_bulk.empty else []
    block_recs = raw_block.to_dict(orient="records") if not raw_block.empty else []

    return bulk_recs, block_recs


# ---------------------------------------------------------------------------
# Strategy 2 — jugaad-data NSELive (fallback)
# ---------------------------------------------------------------------------

def _fetch_deals_jugaad() -> tuple[list[dict], list[dict]]:
    """
    Use ``jugaad_data.nse.NSELive`` to pull deals.

    NSELive may or may not expose bulk/block endpoints depending on version,
    so this is a best-effort fallback.
    """
    from jugaad_data.nse import NSELive  # lazy import

    n = NSELive()

    # Try the method names that different versions may expose
    bulk_recs: list[dict] = []
    block_recs: list[dict] = []

    for method_name, target in [("bulk_deals", bulk_recs), ("block_deals", block_recs)]:
        if hasattr(n, method_name):
            response = getattr(n, method_name)()
            if isinstance(response, dict):
                target.extend(response.get("data", []))
            elif isinstance(response, list):
                target.extend(response)

    if not bulk_recs and not block_recs:
        raise RuntimeError("jugaad-data NSELive has no bulk/block deal methods in this version")

    return bulk_recs, block_recs


# ---------------------------------------------------------------------------
# Strategy 3 — Direct NSE httpx (last resort)
# ---------------------------------------------------------------------------

_NSE_BASE = "https://www.nseindia.com"
_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def _fetch_deals_httpx() -> tuple[list[dict], list[dict]]:
    """
    Hit the NSE bulk/block deals API directly with a cookie-seeded httpx client.
    """
    import httpx

    today_str = datetime.now().strftime("%d-%m-%Y")
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%d-%m-%Y")

    client = httpx.Client(headers=_NSE_HEADERS, follow_redirects=True, timeout=30)
    try:
        # Seed cookies by visiting the homepage
        client.get(_NSE_BASE)

        bulk_resp = client.get(
            f"{_NSE_BASE}/api/historical/bulk-deals",
            params={"from": yesterday_str, "to": today_str},
        )
        bulk_resp.raise_for_status()
        bulk_payload = bulk_resp.json()

        block_resp = client.get(
            f"{_NSE_BASE}/api/historical/block-deals",
            params={"from": yesterday_str, "to": today_str},
        )
        block_resp.raise_for_status()
        block_payload = block_resp.json()
    finally:
        client.close()

    bulk_recs = bulk_payload.get("data", []) if isinstance(bulk_payload, dict) else []
    block_recs = block_payload.get("data", []) if isinstance(block_payload, dict) else []

    return bulk_recs, block_recs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
from agents.hermes_utils import tool

@tool(
    name="get_all_deals",
    description="Fetch all Bulk and Block deals from NSE for the last 24 hours.",
    parameters={
        "type": "object",
        "properties": {},
        "required": []
    },
    toolset="market"
)
def get_all_deals() -> list[dict]:
    """
    Fetch all Bulk and Block deals from NSE for the last 24 hours.

    Tries nselib first, then jugaad-data, then raw httpx as fallback.

    Returns
    -------
    list[dict]
        Each dict has keys: symbol, deal_type, client_name,
        quantity, price, deal_date.
    """
    bulk_recs: list[dict] = []
    block_recs: list[dict] = []
    source = "unknown"

    # --- Attempt 1: nselib ---
    try:
        logger.info("Fetching deals via nselib.capital_market …")
        bulk_recs, block_recs = _fetch_deals_nselib()
        source = "nselib"
        logger.info(
            "nselib returned %d bulk + %d block raw records",
            len(bulk_recs), len(block_recs),
        )
    except Exception as err1:
        logger.warning("nselib failed (%s) — trying jugaad-data fallback", err1)

        # --- Attempt 2: jugaad-data ---
        try:
            bulk_recs, block_recs = _fetch_deals_jugaad()
            source = "jugaad-data"
            logger.info(
                "jugaad-data returned %d bulk + %d block raw records",
                len(bulk_recs), len(block_recs),
            )
        except Exception as err2:
            logger.warning("jugaad-data fallback failed (%s) — trying direct httpx", err2)

            # --- Attempt 3: httpx ---
            try:
                bulk_recs, block_recs = _fetch_deals_httpx()
                source = "httpx"
                logger.info(
                    "httpx returned %d bulk + %d block raw records",
                    len(bulk_recs), len(block_recs),
                )
            except Exception as err3:
                logger.error("All 3 data sources failed. Last error: %s", err3)
                return []

    # Normalise and filter to last 24 hours
    deals: list[dict] = []
    for raw in bulk_recs:
        deal = _normalise(raw, "bulk")
        if _is_within_24h(deal["deal_date"]):
            deals.append(deal)

    for raw in block_recs:
        deal = _normalise(raw, "block")
        if _is_within_24h(deal["deal_date"]):
            deals.append(deal)

    logger.info("Total deals after 24h filter via %s: %d", source, len(deals))
    return deals


@tool(
    name="get_deals_by_symbol",
    description="Return all deals from the last 24 hours for a specific stock ticker.",
    parameters={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "NSE stock ticker (e.g. 'RELIANCE', 'INFY')."
            }
        },
        "required": ["symbol"]
    },
    toolset="market"
)
def get_deals_by_symbol(symbol: str) -> list[dict]:
    """
    Return all deals from the last 24 hours for a specific stock ticker.

    Parameters
    ----------
    symbol : str
        NSE stock ticker (e.g. ``"RELIANCE"``, ``"INFY"``).
        Matching is case-insensitive.

    Returns
    -------
    list[dict]
        Subset of all deals matching the requested symbol.
    """
    symbol_upper = symbol.strip().upper()
    all_deals = get_all_deals()
    filtered = [d for d in all_deals if d["symbol"] == symbol_upper]
    logger.info(
        "get_deals_by_symbol('%s') -> %d deal(s) found", symbol_upper, len(filtered)
    )
    return filtered


def format_deals_for_rag(deals: list[dict]) -> str:
    """
    Convert a list of deal dicts into a human-readable text block suitable
    for injecting into an LLM context window (RAG context).

    Each deal is rendered as a short line so the LLM can reason about
    it naturally.

    Parameters
    ----------
    deals : list[dict]
        Output from ``get_all_deals()`` or ``get_deals_by_symbol()``.

    Returns
    -------
    str
        A formatted multi-line string, or an informational message when
        the list is empty.

    Example output
    --------------
    NSE Bulk & Block Deals (last 24 hours) — 3 deal(s) found:

    [  1] RELIANCE     | BULK  | 250,000 shares @ ₹2,850.00 | Client: HDFC MF | Date: 24-Apr-2026
    [  2] INFY         | BLOCK | 80,000 shares @ ₹1,420.50  | Client: ICICI PRUDENTIAL | Date: 24-Apr-2026
    """
    if not deals:
        return "No NSE bulk/block deals found in the last 24 hours."

    lines = [
        f"NSE Bulk & Block Deals (last 24 hours) — {len(deals)} deal(s) found:\n"
    ]
    for idx, deal in enumerate(deals, start=1):
        # Pretty-format quantity
        try:
            qty_fmt = f"{int(float(deal['quantity'])):,}"
        except (ValueError, TypeError):
            qty_fmt = deal["quantity"]

        # Pretty-format price
        try:
            price_fmt = f"₹{float(deal['price']):,.2f}"
        except (ValueError, TypeError):
            price_fmt = deal["price"]

        line = (
            f"[{idx:>3}] {deal['symbol']:<12} | "
            f"{deal['deal_type'].upper():<5} | "
            f"{qty_fmt} shares @ {price_fmt} | "
            f"Client: {deal['client_name']} | "
            f"Date: {deal['deal_date']}"
        )
        lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick smoke-test — hit real NSE endpoints
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure UTF-8 output on Windows consoles (₹ symbol, etc.)
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print("  Dalal Radar — Market Agent Live Test")
    print("=" * 70)

    print("\n[1] Fetching all deals from the last 24 hours …\n")
    all_deals = get_all_deals()
    print(f"    Found {len(all_deals)} deal(s).\n")

    if all_deals:
        print("[2] RAG-formatted output:\n")
        print(format_deals_for_rag(all_deals))

        # Pick the first symbol and filter
        first_symbol = all_deals[0]["symbol"]
        print(f"\n[3] Deals for symbol '{first_symbol}':\n")
        symbol_deals = get_deals_by_symbol(first_symbol)
        print(format_deals_for_rag(symbol_deals))
    else:
        print("    No deals returned — NSE may have no activity in the last 24h.")
        print("    (Markets may be closed or API is rate-limiting.)")

    print("\n" + "=" * 70)
