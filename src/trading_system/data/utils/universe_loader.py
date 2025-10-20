"""
Utilities for loading a stock universe from CSV files.

KISS: single responsibility - only load and clean symbols from CSV.
Validation of symbols and downstream classification are handled elsewhere.
"""

from __future__ import annotations

import os
import re
import logging
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


logger = logging.getLogger(__name__)


_TICKER_COLUMN = "ticker"
_SECTOR_COLUMNS = ("sector", "section")
_REGION_COLUMNS = ("region", "country_code")
_MARKET_CAP_COLUMNS = ("market_cap", "market_cap_corrected")
_WEIGHT_COLUMNS = ("weight",)


def _first_present(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_ticker(raw: Any) -> Optional[str]:
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    if not s:
        return None
    # Keep alphanumerics, dot and hyphen (common in tickers like BRK.B, RDS-A)
    s = re.sub(r"[^A-Za-z0-9.\-]", "", s)
    s = s.upper()
    return s or None


def _resolve_symbol_minimal(ticker: str, country_code: Optional[str]) -> List[str]:
    """Compose Yahoo-ready symbol candidates from ticker and country/exchange code.

    This is a pure function with a minimal, stable rule set. It keeps `ticker`
    and `country_code` as the single source of truth and avoids per-stock maps.

    Returns ordered candidates to try; the first is the best guess for universe output.
    """
    if not ticker:
        return []

    raw = str(ticker).strip().upper()
    code = str(country_code).strip().upper() if country_code is not None else ""

    # Direct-suffix exchanges: code can be appended directly as .CODE
    direct_suffix = {
        "HK", "L", "TO", "DE", "PA", "MI", "AS", "SW", "VX",
        "KS", "KQ", "AX", "SS", "SZ", "ST", "OL", "VI"
    }

    # Minimal exact mappings (as small as possible to avoid over-engineering)
    map_exact = {
        "UW": "", "UN": "", "US": "",
        "GR": ".DE", "FP": ".PA", "NA": ".AS",
    }

    # Small set of alternates where markets commonly have multiple Yahoo suffixes
    map_alternates = {
        "TT": [".TW", ".TWO"],
        "CN": [".TO", ".V"],  # When source uses CN but actual listing varies
        "SW": [".SW"], "VX": [".VX"],
    }

    candidates: List[str] = []

    if code in direct_suffix:
        candidates.append(f"{raw}.{code}")
    elif code in map_exact:
        candidates.append(f"{raw}{map_exact[code]}")
    elif code in map_alternates:
        for suf in map_alternates[code]:
            candidates.append(f"{raw}{suf}")

    if raw not in candidates:
        candidates.append(raw)

    return candidates


def _apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    result = df

    # Market cap filter
    min_market_cap = filters.get("min_market_cap")
    if min_market_cap is not None:
        mc_col = _first_present(result, _MARKET_CAP_COLUMNS)
        if mc_col is None:
            logger.warning("min_market_cap specified but no market cap column found; skipping this filter")
        else:
            result = result[pd.to_numeric(result[mc_col], errors="coerce") >= float(min_market_cap)]

    # Weight filter
    min_weight = filters.get("min_weight")
    if min_weight is not None:
        w_col = _first_present(result, _WEIGHT_COLUMNS)
        if w_col is None:
            logger.warning("min_weight specified but no weight column found; skipping this filter")
        else:
            result = result[pd.to_numeric(result[w_col], errors="coerce") >= float(min_weight)]

    # Sector include/exclude
    include_sectors = filters.get("include_sectors")
    exclude_sectors = filters.get("exclude_sectors")
    sec_col = _first_present(result, _SECTOR_COLUMNS)
    if include_sectors and sec_col:
        include_set = {str(x).strip().lower() for x in include_sectors}
        result = result[result[sec_col].astype(str).str.strip().str.lower().isin(include_set)]
    if exclude_sectors and sec_col:
        exclude_set = {str(x).strip().lower() for x in exclude_sectors}
        result = result[~result[sec_col].astype(str).str.strip().str.lower().isin(exclude_set)]

    # Regions filter
    regions = filters.get("regions")
    reg_col = _first_present(result, _REGION_COLUMNS)
    if regions and reg_col:
        regions_set = {str(x).strip().upper() for x in regions}
        result = result[result[reg_col].astype(str).str.strip().str.upper().isin(regions_set)]

    return result


def _limit_max_stocks(df: pd.DataFrame, max_stocks: Optional[int]) -> pd.DataFrame:
    if not max_stocks or max_stocks <= 0:
        return df
    # Prefer sort by weight desc, else market cap desc, else keep current order
    w_col = _first_present(df, _WEIGHT_COLUMNS)
    if w_col is not None:
        sorted_df = df.copy()
        sorted_df[w_col] = pd.to_numeric(sorted_df[w_col], errors="coerce")
        sorted_df = sorted_df.sort_values(by=w_col, ascending=False, kind="mergesort")
        return sorted_df.head(int(max_stocks))

    mc_col = _first_present(df, _MARKET_CAP_COLUMNS)
    if mc_col is not None:
        sorted_df = df.copy()
        sorted_df[mc_col] = pd.to_numeric(sorted_df[mc_col], errors="coerce")
        sorted_df = sorted_df.sort_values(by=mc_col, ascending=False, kind="mergesort")
        return sorted_df.head(int(max_stocks))

    # As-is (stable head)
    return df.head(int(max_stocks))


def load_universe_from_csv(csv_path: str, filters: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Load a stock universe from a CSV file and return a cleaned list of symbols.

    Args:
        csv_path: Path to the CSV file.
        filters: Optional filtering options:
            - min_market_cap (float): Minimum market cap in billions USD
            - min_weight (float): Minimum weight threshold
            - max_stocks (int): Maximum number of stocks to return
            - include_sectors (List[str]): Sectors to include
            - exclude_sectors (List[str]): Sectors to exclude
            - regions (List[str]): Region whitelist (e.g., ["US", "EU"]). Compared against region/country_code

    Returns:
        List[str]: Cleaned list of ticker symbols.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If format invalid (missing ticker) or filtered result is empty.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV '{csv_path}': {e}")

    if _TICKER_COLUMN not in df.columns:
        raise ValueError("CSV missing required 'ticker' column")

    filters = filters or {}

    # Apply filters (operate on a copy to avoid SettingWithCopy issues)
    filtered = _apply_filters(df.copy(), filters)

    # Limit number of stocks if requested
    filtered = _limit_max_stocks(filtered, filters.get("max_stocks"))

    # Clean and deduplicate symbols while preserving order
    # Prefer composing Yahoo-ready symbols from ticker + country_code when available
    tickers: List[str] = []
    seen = set()

    country_col = "country_code" if "country_code" in filtered.columns else None

    for _, row in filtered.iterrows():
        raw_ticker = row.get(_TICKER_COLUMN)
        t = _normalize_ticker(raw_ticker)
        if not t:
            continue

        composed = None
        if country_col:
            candidates = _resolve_symbol_minimal(t, row.get(country_col))
            if candidates:
                composed = candidates[0]
        final_symbol = composed or t

        if final_symbol in seen:
            continue
        seen.add(final_symbol)
        tickers.append(final_symbol)

    if not tickers:
        raise ValueError("No tickers after applying filters and cleaning; consider relaxing filter conditions")

    logger.info("Loaded %d tickers from '%s'", len(tickers), csv_path)
    return tickers


def load_symbols_from_config(config: Dict[str, Any]) -> List[str]:
    """
    Resolve symbols according to Option A config integration.

    Prefers `training_setup.parameters.universe` when present. Falls back to
    `training_setup.parameters.symbols` when provided and non-empty.
    """
    params = (config or {}).get("training_setup", {}).get("parameters", {})
    uni = params.get("universe")
    if uni and uni.get("source") == "csv":
        csv_path = uni.get("csv_path")
        if not csv_path:
            raise ValueError("universe.source=csv specified but csv_path is missing")
        return load_universe_from_csv(csv_path, uni.get("filters", {}))

    syms = params.get("symbols")
    if syms:
        return list(syms)

    raise ValueError("No symbols or universe configuration found in training_setup.parameters")


class UniverseProvider:
    """
    Thin resolver class for obtaining a training universe (symbols list)
    from a unified experiment config. Keeps providers decoupled from
    symbol source concerns (CSV vs inline list).
    """

    @staticmethod
    def resolve_symbols(config: Dict[str, Any]) -> List[str]:
        return load_symbols_from_config(config)


