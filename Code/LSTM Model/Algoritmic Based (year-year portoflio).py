

from __future__ import annotations
import sys, os
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm
from pathlib import Path
import json
import math
import numbers


# =============================================================================
# Configuration
# =============================================================================
START_CASH = 1000
DATE_COL = "datetime"
PRICE_COL = "close"
FEE_FIXED_COL = "fee_usd"
FEE_VAR_COL = "trading_fee_pct"

FORECAST_ROOT = "monthly_forecasts_yearly_portfolio"
OUTPUT_ROOT = "monthly_forecasts_Yearly_portfolio"


# =============================================================================
# Parameter Grid and Key Order
# =============================================================================
# Random grid for buy/sell thresholds, slope gates, and cooldowns.
PARAMETER_GRID = {
    "short_gain_mult": [0, 0.2, 0.4, 0.6, 0.8, 1],
    "mid_gain_mult": [0, 0.2, 0.4, 0.6, 0.8, 1],
    "long_gain_mult": [0, 0.2, 0.4, 0.6, 0.8, 1],
    "bullish_slope_short": [0, 0.2, 0.4, 0.6, 0.8, 1],
    "bullish_slope_mid": [0, 0.2, 0.4, 0.6, 0.8, 1],
    "bullish_slope_long": [0, 0.2, 0.4, 0.6, 0.8, 1],
    "short_gain_mult_sell": [0, 0.2, 0.4, 0.6, 0.8, 1],
    "mid_gain_mult_sell": [0, 0.2, 0.4, 0.6, 0.8, 1],
    "long_gain_mult_sell": [0, 0.2, 0.4, 0.6, 0.8, 1],
    "bearish_slope_short": [0, 0.2, 0.4, 0.6, 0.8, 1],
    "bearish_slope_mid": [0, 0.2, 0.4, 0.6, 0.8, 1],
    "bearish_slope_long": [0, 0.2, 0.4, 0.6, 0.8, 1],
    "buy_cooldown_minutes": [20, 30, 40, 50],
    "sell_cooldown_minutes": [20, 30, 40, 50],
}

# Fixed order for fast packing into arrays used by the JIT kernel.
PARAM_KEYS = [
    "short_gain_mult",
    "mid_gain_mult",
    "long_gain_mult",

    "bullish_slope_short",
    "bullish_slope_mid",
    "bullish_slope_long",

    "short_gain_mult_sell",
    "mid_gain_mult_sell",
    "long_gain_mult_sell",

    "bearish_slope_short",
    "bearish_slope_mid",
    "bearish_slope_long",

    "buy_cooldown_minutes",
    "sell_cooldown_minutes",
]

# Forecast columns expected in monthly forecast CSVs.
forecast_cols = [
    "forecast_t+1m", "forecast_t+2m", "forecast_t+5m", "forecast_t+10m",
    "forecast_t+30m", "forecast_t+60m", "forecast_t+300m", "forecast_t+600m",
    "forecast_t+1440m", "forecast_t+2880m", "forecast_t+7200m", "forecast_t+10080m",
]

# CSV read schema (subset for efficiency and dtype control).
USECOLS = [DATE_COL, PRICE_COL, FEE_FIXED_COL, FEE_VAR_COL] + forecast_cols


# =============================================================================
# Utilities
# =============================================================================


def app_dir() -> str:
    """Folder that contains the .exe when frozen, or this .py when not."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = app_dir()

# Make your roots absolute, anchored to the app folder
FORECAST_ROOT_ABS = os.path.join(BASE_DIR, FORECAST_ROOT)
OUTPUT_ROOT_ABS   = os.path.join(BASE_DIR, OUTPUT_ROOT)

def load_df(csv_path: str) -> pd.DataFrame:
    """Load a forecast-month CSV with required columns and dtypes."""
    df = pd.read_csv(
        csv_path,
        usecols=USECOLS,
        memory_map=True,
        dtype={
            PRICE_COL: "float32",
            FEE_FIXED_COL: "float32",
            FEE_VAR_COL: "float32",
            **{c: "float32" for c in forecast_cols},
        },
    )
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df

def _params_to_array(params: dict) -> np.ndarray:
    params = _normalize_params(params)
    return np.array([params[k] for k in PARAM_KEYS], dtype=np.float64)

def _to_py_scalar(x):
    """Convert NumPy scalars to native Python types for JSON serialization."""
    if isinstance(x, np.generic):
        return x.item()
    return x

def _normalize_params(params: dict) -> dict:
    """Cast parameters to Python types; ints for cooldowns, floats otherwise."""
    out = {}
    for k in PARAM_KEYS:
        v = _to_py_scalar(params[k])
        if "cooldown" in k:
            out[k] = int(v)
        else:
            out[k] = float(v)
    return out

def _json_default(o):
    """Fallback JSON encoder for stray NumPy types."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, numbers.Number):
        return float(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

def _evaluate_month_value_with_jit(df: pd.DataFrame, params: dict) -> float:
    """Evaluate terminal portfolio value on a month via the JIT kernel."""
    closes, fee_var, fee_fixed, futures, sp, mp, lp = precompute_features(df)
    arr = _params_to_array(params)
    return float(evaluate_params(arr, closes, fee_var, fee_fixed, futures, sp, mp, lp))

def _load_champion(champion_csv_path: str) -> dict | None:
    """Load the global/per-month champion row from CSV, if present."""
    if not os.path.exists(champion_csv_path):
        return None
    df = pd.read_csv(champion_csv_path)
    if df.empty:
        return None
    row = df.iloc[0].to_dict()
    params = json.loads(row["params"])
    row["params"] = params
    return row

def _save_champion(params: dict, metric: float, month_label: str, source: str, champion_csv_path: str) -> None:
    """Persist the selected parameter set (with metadata) to CSV."""
    params = _normalize_params(params)
    row = {
        "params": json.dumps(params, separators=(",", ":"), default=_json_default),
        "metric": float(metric) if metric is not None else float("nan"),
        "evaluated_on_month": month_label,
        "source": source,
    }
    pd.DataFrame([row]).to_csv(champion_csv_path, index=False)

def _higher_is_better(a: float, b: float) -> bool:
    """Comparison helper: treat NaNs as inferior."""
    if a is None or math.isnan(a):
        return False
    if b is None or math.isnan(b):
        return True
    return a > b


# =============================================================================
# Feature Precomputation (for the JIT evaluator)
# =============================================================================
_feature_cache: dict[int, tuple] = {}

def precompute_features(df: pd.DataFrame):
    """Cache arrays needed by the numba kernel for a given DataFrame."""
    key = id(df)
    if key in _feature_cache:
        return _feature_cache[key]

    closes = df[PRICE_COL].values
    fee_var = df[FEE_VAR_COL].values if FEE_VAR_COL in df else np.zeros_like(closes)
    fee_fixed = df[FEE_FIXED_COL].values if FEE_FIXED_COL in df else np.zeros_like(closes)
    futures = np.stack([df[c].values for c in forecast_cols], axis=1)

    # Past-price windows (left-padded to keep first windows full)
    pad = 12
    padded = np.concatenate([np.full(pad, closes[0], dtype=closes.dtype), closes])
    sp = np.lib.stride_tricks.sliding_window_view(padded, 3)[pad : pad + len(closes)]
    mp = np.lib.stride_tricks.sliding_window_view(padded, 6)[pad : pad + len(closes)]
    lp = np.lib.stride_tricks.sliding_window_view(padded, 11)[pad : pad + len(closes)]

    _feature_cache[key] = (closes, fee_var, fee_fixed, futures, sp, mp, lp)
    return _feature_cache[key]


# =============================================================================
# JIT-Compiled Evaluator
# =============================================================================
@njit
def evaluate_params(params_arr: np.ndarray,
                    closes: np.ndarray,
                    fee_var: np.ndarray,
                    fee_fixed: np.ndarray,
                    futures: np.ndarray,
                    sp: np.ndarray,
                    mp: np.ndarray,
                    lp: np.ndarray) -> float:
    """Simulate one month with a given parameter array and return terminal value.

    Uses least-squares slope over min–max normalized past+future windows
    (consistent with the vectorized simulator). Trades incur fixed and
    proportional fees. Cooldowns prevent immediate re-entry/exit.
    """
    cash = START_CASH
    btc = 0.0
    n = len(closes)

    buy_cd = int(params_arr[12])
    sell_cd = int(params_arr[13])
    last_buy_i = -10_000_000
    last_sell_i = -10_000_000

    # Local helper for normalized slope across concatenated windows
    def slope_normed(past_view, future_view):
        Lp = past_view.shape[0]
        Lf = future_view.shape[0]
        L = Lp + Lf
        if L < 3:
            return 0.0

        # Min/max over both segments
        y_min = 1.0e308
        y_max = -1.0e308
        for j in range(Lp):
            v = float(past_view[j])
            if v < y_min:
                y_min = v
            if v > y_max:
                y_max = v
        for k in range(Lf):
            v = float(future_view[k])
            if v < y_min:
                y_min = v
            if v > y_max:
                y_max = v
        rng = y_max - y_min
        if rng == 0.0:
            rng = 1e-8

        # Precompute Σx and Σx² for x=0..L-1
        N = float(L)
        sum_x  = (L - 1) * L / 2.0
        sum_x2 = (L - 1) * L * (2 * L - 1) / 6.0

        sum_y = 0.0
        sum_xy = 0.0

        # Past part
        for j in range(Lp):
            y = (float(past_view[j]) - y_min) / rng
            sum_y += y
            sum_xy += j * y

        # Future part
        for k in range(Lf):
            x = Lp + k
            y = (float(future_view[k]) - y_min) / rng
            sum_y += y
            sum_xy += x * y

        denom = N * sum_x2 - (sum_x * sum_x)
        m = 0.0 if denom == 0.0 else (N * sum_xy - sum_x * sum_y) / denom
        return m * (L - 1)

    for i in range(n):
        price = closes[i]
        fee_var_i = fee_var[i]
        fee_fixed_i = fee_fixed[i]
        fut = futures[i]

        # Slopes (3/6/11 past vs 4/7/12 future), normalized
        S_s = slope_normed(sp[i],  fut[:4])
        S_m = slope_normed(mp[i],  fut[:7])
        S_l = slope_normed(lp[i],  fut[:12])

        # Expected gains vs current price
        g0 = float(np.max(fut[:4])  - price)
        g1 = float(np.max(fut[:7])  - price)
        g2 = float(np.max(fut[:12]) - price)
        cost = 2.0 * fee_var_i * price + fee_fixed_i

        # Buy: only when flat and cooldown since last sell have passed
        if cash > 0.0 and (i - last_sell_i) >= buy_cd:
            if g0 > params_arr[0] * cost and S_s > params_arr[3]:
                btc = cash / (price * (1.0 + fee_var_i) + fee_fixed_i)
                cash = 0.0
                last_buy_i = i
            elif g1 > params_arr[1] * cost and S_m > params_arr[4]:
                btc = cash / (price * (1.0 + fee_var_i) + fee_fixed_i)
                cash = 0.0
                last_buy_i = i
            elif g2 > params_arr[2] * cost and S_l > params_arr[5]:
                btc = cash / (price * (1.0 + fee_var_i) + fee_fixed_i)
                cash = 0.0
                last_buy_i = i

        # Sell: only when long and cooldown since last buy have passed
        if btc > 0.0 and (i - last_buy_i) >= sell_cd:
            if g0 < -params_arr[6] * cost and S_s < -params_arr[9]:
                cash = btc * price * (1.0 - fee_var_i) - fee_fixed_i
                btc = 0.0
                last_sell_i = i
            elif g1 < -params_arr[7] * cost and S_m < -params_arr[10]:
                cash = btc * price * (1.0 - fee_var_i) - fee_fixed_i
                btc = 0.0
                last_sell_i = i
            elif g2 < -params_arr[8] * cost and S_l < -params_arr[11]:
                cash = btc * price * (1.0 - fee_var_i) - fee_fixed_i
                btc = 0.0
                last_sell_i = i

    return cash + btc * closes[-1]

# Warm up compilation reference (kept for parity with original design)
_dummy = evaluate_params.py_func  # noqa: F841


# =============================================================================
# Grid Search
# =============================================================================

def run_grid_search(csv_path: str, config_path: str, n_iter: int = 3000, random_state: int = 50):
    """Randomly sample parameter combinations; cache and return the best.

    Returns
    -------
    (best_params, best_value, was_cached)
    """
    rel = os.path.relpath(os.path.dirname(csv_path), FORECAST_ROOT)
    out_dir = os.path.join(OUTPUT_ROOT_ABS, rel)
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, "best_params.csv")

    if os.path.exists(config_path):
        dfc = pd.read_csv(config_path)
        best_row = dfc.sort_values("final_value", ascending=False).iloc[0]
        best = best_row.drop(labels=["final_value"]).to_dict()
        best = _normalize_params(best)
        best_value = float(best_row["final_value"])
        print(f"[GRID] Loaded cached params for {rel}")
        return best, best_value, True

    print(f"[GRID] Running grid search for {rel} ({n_iter} samples)")
    df = load_df(csv_path)
    df.set_index(DATE_COL, inplace=True)
    closes, fee_var, fee_fixed, futures, sp, mp, lp = precompute_features(df)

    rng = np.random.default_rng()

    def random_param_sampler(grid, n, rng):
        keys = list(grid.keys())
        for _ in range(n):
            yield {k: rng.choice(grid[k]) for k in keys}

    sampler = list(random_param_sampler(PARAMETER_GRID, n_iter, rng))

    results = []
    for _, params in enumerate(tqdm(sampler, desc="Grid")):
        arr = np.array([params[k] for k in PARAM_KEYS], dtype=np.float64)
        val = evaluate_params(arr, closes, fee_var, fee_fixed, futures, sp, mp, lp)
        results.append((val, params))

    results.sort(key=lambda x: x[0], reverse=True)
    full = pd.DataFrame([{"final_value": v, **p} for v, p in results])
    full.to_csv(config_path, index=False)
    print(f"[GRID] Best value for {rel}: {results[0][0]:.2f}")
    return results[0][1], float(results[0][0]), False


# =============================================================================
# Historical Simulation (Vectorized)
# =============================================================================

def _slope_normalized_window(past_view: np.ndarray, future_view: np.ndarray) -> float:
    """Least-squares slope on min–max normalized concatenated series.

    Returns slope * (L-1) to match the JIT evaluator's scale.
    """
    Lp = past_view.shape[0]
    Lf = future_view.shape[0]
    L = Lp + Lf
    if L < 3:
        return 0.0

    y_min = float("inf")
    y_max = float("-inf")
    if Lp:
        pv_min = float(past_view.min()); pv_max = float(past_view.max())
        if pv_min < y_min: y_min = pv_min
        if pv_max > y_max: y_max = pv_max
    if Lf:
        fv_min = float(future_view.min()); fv_max = float(future_view.max())
        if fv_min < y_min: y_min = fv_min
        if fv_max > y_max: y_max = fv_max

    rng = y_max - y_min
    if rng == 0.0:
        rng = 1e-8

    # Precompute Σx and Σx² for x=0..L-1
    sum_x = (L - 1) * L / 2.0
    sum_x2 = (L - 1) * L * (2 * L - 1) / 6.0

    sum_y = 0.0
    sum_xy = 0.0

    for j in range(Lp):
        y = (float(past_view[j]) - y_min) / rng
        sum_y += y
        sum_xy += j * y

    for k in range(Lf):
        x = Lp + k
        y = (float(future_view[k]) - y_min) / rng
        sum_y += y
        sum_xy += x * y

    N = float(L)
    denom = N * sum_x2 - (sum_x * sum_x)
    m = 0.0 if denom == 0.0 else (N * sum_xy - sum_x * sum_y) / denom
    return m * (L - 1)


def simulate_history_fast(df: pd.DataFrame,
                          params: dict,
                          forecast_cols: list[str],
                          cash0: float = START_CASH,
                          btc0: float = 0.0) -> pd.DataFrame:
    """Vectorized month simulation returning portfolio trajectory and fees.

    Starts from the provided (cash0, btc0) to support year-to-date carry-over.
    """
    idx = df.index

    closes = df[PRICE_COL].to_numpy(dtype=float)
    fee_var = (
        df[FEE_VAR_COL].to_numpy(dtype=float) if FEE_VAR_COL in df.columns else np.zeros(len(df), dtype=float)
    )
    fee_fixed = (
        df[FEE_FIXED_COL].to_numpy(dtype=float) if FEE_FIXED_COL in df.columns else np.zeros(len(df), dtype=float)
    )
    futures = df[forecast_cols].to_numpy(dtype=float)

    n, _ = futures.shape
    P_SHORT, F_SHORT = 3, 4
    P_MID, F_MID = 6, 7
    P_LONG, F_LONG = 11, 12

    # Extract parameters once
    short_gain_mult = float(params["short_gain_mult"]) ; mid_gain_mult = float(params["mid_gain_mult"]) ; long_gain_mult = float(params["long_gain_mult"]) 
    bullish_slope_short = float(params["bullish_slope_short"]) ; bullish_slope_mid = float(params["bullish_slope_mid"]) ; bullish_slope_long = float(params["bullish_slope_long"]) 
    short_gain_mult_sell = float(params["short_gain_mult_sell"]) ; mid_gain_mult_sell = float(params["mid_gain_mult_sell"]) ; long_gain_mult_sell = float(params["long_gain_mult_sell"]) 
    bearish_slope_short = float(params["bearish_slope_short"]) ; bearish_slope_mid = float(params["bearish_slope_mid"]) ; bearish_slope_long = float(params["bearish_slope_long"]) 
    buy_cd = int(params.get("buy_cooldown_minutes", 0))
    sell_cd = int(params.get("sell_cooldown_minutes", 0))

    # Initialize state from inputs
    cash = float(cash0)
    btc = float(btc0)
    last_buy_i = -10_000_000
    last_sell_i = -10_000_000

    out = np.empty((n, 7), dtype=float)

    for i in range(n):
        price = closes[i]
        fv = fee_var[i]
        ff = fee_fixed[i]
        fut = futures[i]

        fee_fixed_tick = 0.0
        fee_var_tick = 0.0

        if np.any(np.isnan(fut)):
            out[i, 0] = cash; out[i, 1] = btc; out[i, 2] = price
            out[i, 3] = cash + btc * price; out[i, 4] = 0.0; out[i, 5] = 0.0; out[i, 6] = 0.0
            continue

        # Slopes (no temporary concatenation)
        p_len = min(P_SHORT, i); past_view = closes[i - p_len : i] if p_len > 0 else closes[0:0]; S_s = _slope_normalized_window(past_view, fut[:F_SHORT])
        p_len = min(P_MID, i);   past_view = closes[i - p_len : i] if p_len > 0 else closes[0:0]; S_m = _slope_normalized_window(past_view, fut[:F_MID])
        p_len = min(P_LONG, i);  past_view = closes[i - p_len : i] if p_len > 0 else closes[0:0]; S_l = _slope_normalized_window(past_view, fut[:F_LONG])

        g0 = float(np.max(fut[:F_SHORT]) - price)
        g1 = float(np.max(fut[:F_MID]) - price)
        g2 = float(np.max(fut[:F_LONG]) - price)
        cost = 2.0 * fv * price + ff

        traded = False

        # Buy
        if cash > 0.0 and (i - last_sell_i) >= buy_cd:
            if g0 > short_gain_mult * cost and S_s > bullish_slope_short:
                btc_new = cash / (price * (1.0 + fv) + ff)
                fee_fixed_tick += ff; fee_var_tick += btc_new * price * fv
                btc = btc_new; cash = 0.0; last_buy_i = i; traded = True
            elif g1 > mid_gain_mult * cost and S_m > bullish_slope_mid:
                btc_new = cash / (price * (1.0 + fv) + ff)
                fee_fixed_tick += ff; fee_var_tick += btc_new * price * fv
                btc = btc_new; cash = 0.0; last_buy_i = i; traded = True
            elif g2 > long_gain_mult * cost and S_l > bullish_slope_long:
                btc_new = cash / (price * (1.0 + fv) + ff)
                fee_fixed_tick += ff; fee_var_tick += btc_new * price * fv
                btc = btc_new; cash = 0.0; last_buy_i = i; traded = True

        # Sell
        if btc > 0.0 and (i - last_buy_i) >= sell_cd:
            if g0 < -short_gain_mult_sell * cost and S_s < -bearish_slope_short:
                fee_fixed_tick += ff; fee_var_tick += btc * price * fv
                cash = btc * price * (1.0 - fv) - ff; btc = 0.0; last_sell_i = i; traded = True
            elif g1 < -mid_gain_mult_sell * cost and S_m < -bearish_slope_mid:
                fee_fixed_tick += ff; fee_var_tick += btc * price * fv
                cash = btc * price * (1.0 - fv) - ff; btc = 0.0; last_sell_i = i; traded = True
            elif g2 < -long_gain_mult_sell * cost and S_l < -bearish_slope_long:
                fee_fixed_tick += ff; fee_var_tick += btc * price * fv
                cash = btc * price * (1.0 - fv) - ff; btc = 0.0; last_sell_i = i; traded = True

        out[i, 0] = cash
        out[i, 1] = btc
        out[i, 2] = price
        out[i, 3] = cash + btc * price
        out[i, 4] = 0.0
        out[i, 5] = fee_fixed_tick
        out[i, 6] = fee_var_tick

    return pd.DataFrame(
        {
            "datetime": idx,
            "cash": out[:, 0],
            "btc": out[:, 1],
            "price": out[:, 2],
            "portfolio_value": out[:, 3],
            "fixed_fee_col": out[:, 5],
            "fee_var_col": out[:, 6],
        },
        index=idx,
    )


# =============================================================================
# Month Simulation Wrapper
# =============================================================================

def simulate_month_pair(pair: tuple[str, str]) -> None:
    """Grid on month d0, pick winner vs champion, simulate month d1, save artifacts."""
    d0, d1 = pair

    # Next-month outputs (per-month artifacts)
    rel_next = os.path.relpath(d1, FORECAST_ROOT)
    out_dir_next = os.path.join(OUTPUT_ROOT_ABS, rel_next)
    os.makedirs(out_dir_next, exist_ok=True)
    champion_csv_month_next = os.path.join(out_dir_next, "best_params_live.csv")

    # Global champion under monthly_forecasts/
    base = BASE_DIR
    forecast_root_abs = FORECAST_ROOT_ABS
    os.makedirs(forecast_root_abs, exist_ok=True)
    champion_csv_global = os.path.join(forecast_root_abs, "best_params_live.csv")

    # Grid-search on CURRENT month (d0)
    forecast0 = os.path.join(d0, f"forecast_{os.path.basename(d0)}.csv")
    cfg = os.path.join(out_dir_next, "best_params.csv")
    grid_best_params, grid_best_value, _ = run_grid_search(forecast0, cfg)

    # Evaluate GLOBAL champion on CURRENT month (d0)
    df0 = load_df(forecast0); df0.set_index(DATE_COL, inplace=True)
    current_month_label = os.path.basename(d0)

    champion_row = _load_champion(champion_csv_global)  # load only global
    champ_value = float("nan")
    if champion_row is not None:
        try:
            champ_value = _evaluate_month_value_with_jit(df0, champion_row["params"])
        except Exception as e:
            print(f"[champion-eval] failed on {current_month_label}: {e}")

    # Pick winner (higher is better; NaN loses)
    if _higher_is_better(champ_value, grid_best_value):
        chosen_params = champion_row["params"]; chosen_metric = champ_value; chosen_source = "champion"
        print(f"[select] champion wins on {current_month_label}: {champ_value:.2f} > grid {grid_best_value:.2f}")
    else:
        chosen_params = grid_best_params; chosen_metric = grid_best_value; chosen_source = "grid"
        print(f"[select] grid wins on {current_month_label}: {grid_best_value:.2f} (champion {champ_value})")

    chosen_params = _normalize_params(chosen_params)

    # Save winner to GLOBAL + per-month (next) and verify global write
    _save_champion(chosen_params, chosen_metric, current_month_label, chosen_source, champion_csv_global)
    _save_champion(chosen_params, chosen_metric, current_month_label, chosen_source, champion_csv_month_next)

    # Safety check: read back global and assert match
    readback = _load_champion(champion_csv_global)
    if not readback or _params_to_array(readback["params"]).tolist() != _params_to_array(chosen_params).tolist():
        print("[WARN] Global champion readback mismatch — forcing overwrite once more.")
        _save_champion(chosen_params, chosen_metric, current_month_label, chosen_source, champion_csv_global)

    # Simulate NEXT month (d1) with chosen params
    forecast1 = os.path.join(d1, f"forecast_{os.path.basename(d1)}.csv")
    df1 = load_df(forecast1); df1.set_index(DATE_COL, inplace=True)
    history = simulate_history_fast(df1, chosen_params, forecast_cols, cash0=START_CASH)

    # Buy & Hold baseline and outputs
    fee_var0 = df1[FEE_VAR_COL].iloc[0] if FEE_VAR_COL in df1.columns else 0.0
    fee_fixed0 = df1[FEE_FIXED_COL].iloc[0] if FEE_FIXED_COL in df1.columns else 0.0
    btc0 = START_CASH / (history["price"].iloc[0] * (1.0 + fee_var0) + fee_fixed0)
    history["buy_hold_value"] = history["price"] * btc0

    history.to_csv(os.path.join(out_dir_next, "portfolio.csv"), index=False)
    plt.figure(figsize=(12, 6))
    plt.plot(history["datetime"], history["portfolio_value"], label="Algorithmic-based")
    plt.plot(history["datetime"], history["buy_hold_value"], "--", label="Buy & Hold")
    plt.legend(); plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(os.path.join(out_dir_next, "portfolio.png"))
    plt.close()
    print(f"[SIM] Completed {os.path.relpath(d1, FORECAST_ROOT)}, final={history['portfolio_value'].iloc[-1]:.2f}")


# =============================================================================
# Year Walker
# =============================================================================

def simulate_year(months_for_year: list[tuple[int, int, str]]) -> None:
    """Simulate a sequence of month pairs within a single year.

    The portfolio resets at the start of each year. Buy & Hold is computed once
    per year using the first month’s initial price and fees.
    """
    year = months_for_year[0][0]

    # Yearly resets
    carry_cash = float(START_CASH)
    carry_btc = 0.0
    buy_hold_btc = None
    year_histories: list[pd.DataFrame] = []

    # Global champion path
    base = BASE_DIR
    forecast_root_abs = FORECAST_ROOT_ABS
    os.makedirs(forecast_root_abs, exist_ok=True)
    champion_csv_global = os.path.join(forecast_root_abs, "best_params_live.csv")

    # Iterate (train-month -> next-month)
    for (_, _, d0), (_, _, d1) in zip(months_for_year, months_for_year[1:]):
        rel_next = os.path.relpath(d1, FORECAST_ROOT)
        out_dir_next = os.path.join(os.getcwd(), OUTPUT_ROOT, rel_next)
        os.makedirs(out_dir_next, exist_ok=True)
        champion_csv_month_next = os.path.join(out_dir_next, "best_params_live.csv")

        forecast0 = os.path.join(d0, f"forecast_{os.path.basename(d0)}.csv")
        cfg = os.path.join(out_dir_next, "best_params.csv")
        grid_best_params, grid_best_value, _ = run_grid_search(forecast0, cfg)

        df0 = load_df(forecast0); df0.set_index(DATE_COL, inplace=True)
        current_month_label = os.path.basename(d0)

        champion_row = _load_champion(champion_csv_global)
        champ_value = float("nan")
        if champion_row is not None:
            try:
                champ_value = _evaluate_month_value_with_jit(df0, champion_row["params"])
            except Exception as e:
                print(f"[champion-eval] failed on {current_month_label}: {e}")

        if _higher_is_better(champ_value, grid_best_value):
            chosen_params = champion_row["params"]; chosen_metric = champ_value; chosen_source = "champion"
            print(f"[select] champion wins on {current_month_label}: {champ_value:.2f} > grid {grid_best_value:.2f}")
        else:
            chosen_params = grid_best_params; chosen_metric = grid_best_value; chosen_source = "grid"
            print(f"[select] grid wins on {current_month_label}: {grid_best_value:.2f} (champion {champ_value})")

        chosen_params = _normalize_params(chosen_params)

        _save_champion(chosen_params, chosen_metric, current_month_label, chosen_source, champion_csv_global)
        _save_champion(chosen_params, chosen_metric, current_month_label, chosen_source, champion_csv_month_next)

        readback = _load_champion(champion_csv_global)
        if not readback or _params_to_array(readback["params"]).tolist() != _params_to_array(chosen_params).tolist():
            print("[WARN] Global champion readback mismatch — forcing overwrite once more.")
            _save_champion(chosen_params, chosen_metric, current_month_label, chosen_source, champion_csv_global)

        forecast1 = os.path.join(d1, f"forecast_{os.path.basename(d1)}.csv")
        df1 = load_df(forecast1); df1.set_index(DATE_COL, inplace=True)
        history = simulate_history_fast(df1, chosen_params, forecast_cols, cash0=carry_cash, btc0=carry_btc)
        history = history.copy()
        history["year"] = year
        # history["month"] = int(os.path.basename(d1)[5:7])  # optional: include month number
        year_histories.append(history)

        if buy_hold_btc is None:
            fee_var0 = df1[FEE_VAR_COL].iloc[0] if FEE_VAR_COL in df1.columns else 0.0
            fee_fixed0 = df1[FEE_FIXED_COL].iloc[0] if FEE_FIXED_COL in df1.columns else 0.0
            buy_hold_btc = START_CASH / (history["price"].iloc[0] * (1.0 + fee_var0) + fee_fixed0)
        history["buy_hold_value"] = history["price"] * buy_hold_btc

        out_dir_next = os.path.join(os.getcwd(), OUTPUT_ROOT, os.path.relpath(d1, FORECAST_ROOT))
        os.makedirs(out_dir_next, exist_ok=True)
        history.to_csv(os.path.join(out_dir_next, "portfolio.csv"), index=False)
        plt.figure(figsize=(12, 6))
        plt.plot(history["datetime"], history["portfolio_value"], label="Algorithmic-based")
        plt.plot(history["datetime"], history["buy_hold_value"], "--", label="Buy & Hold")
        plt.legend(); plt.xticks(rotation=45); plt.tight_layout()
        plt.savefig(os.path.join(out_dir_next, "portfolio.png"))
        plt.close()

        # Carry state into the next month
        carry_cash = float(history["cash"].iloc[-1])
        carry_btc  = float(history["btc"].iloc[-1])

        print(f"[SIM] Completed {os.path.relpath(d1, FORECAST_ROOT)}, final={history['portfolio_value'].iloc[-1]:.2f}")

    if year_histories:
            year_history = pd.concat(year_histories, ignore_index=True)

            # Output directory: yearly_portfolio_results/<year>/

            yearly_root = OUTPUT_ROOT_ABS
            year_dir = os.path.join(yearly_root, str(year))
            os.makedirs(year_dir, exist_ok=True)

            # Save the full-year time series
            csv_path = os.path.join(year_dir, "portfolio_year.csv")
            year_history.to_csv(csv_path, index=False)

            # Plot full-year portfolio vs Buy & Hold
            plt.figure(figsize=(14, 6))
            plt.plot(year_history["datetime"], year_history["portfolio_value"], label="Algorithmic-based")
            if "buy_hold_value" in year_history:
                plt.plot(year_history["datetime"], year_history["buy_hold_value"], "--", label="Buy & Hold")
            plt.title(f"{year} Portfolio (yearly reset)")
            plt.xlabel("Time"); plt.ylabel("Value")
            plt.legend(); plt.xticks(rotation=45); plt.tight_layout()
            png_path = os.path.join(year_dir, "portfolio_year.png")
            plt.savefig(png_path)
            plt.close()

            # Append a compact yearly summary row
            final_row = year_history.iloc[-1]
            summary_path = os.path.join(yearly_root, "yearly_summary.csv")
            header_needed = not os.path.exists(summary_path)
            with open(summary_path, "a") as f:
                if header_needed:
                    f.write("year,final_value,final_cash,final_btc,first_ts,last_ts\n")
                f.write(
                    f"{year},{final_row['portfolio_value']:.10g},{final_row['cash']:.10g},"
                    f"{final_row['btc']:.10g},{year_history['datetime'].iloc[0]},{final_row['datetime']}\n"
                )

            print(f"[YEAR] {year}: saved CSV -> {os.path.relpath(csv_path)} and plot -> {os.path.relpath(png_path)}")

def month_dirs(root: str):
    """Return (year, month, path) tuples for subfolders named 'Month_YYYY'."""
    months = []
    for yr in sorted(os.listdir(root)):
        yp = os.path.join(root, yr)
        if not os.path.isdir(yp):
            continue
        for mon in sorted(os.listdir(yp)):
            mp = os.path.join(yp, mon)
            if not os.path.isdir(mp):
                continue
            m_str, y_str = mon.split("_")
            m = list(calendar.month_name).index(m_str.capitalize())
            y = int(y_str)
            months.append((y, m, mp))
    return sorted(months, key=lambda x: (x[0], x[1]))


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    base = BASE_DIR
    forecast_root = FORECAST_ROOT_ABS
    if not os.path.isdir(forecast_root):
        raise FileNotFoundError(f"Missing {FORECAST_ROOT}")
    months = month_dirs(forecast_root)

    # Group by year; for each year, run month->month sequence with carry-over
    from itertools import groupby
    for year, group in groupby(months, key=lambda t: t[0]):
        year_months = list(group)
        if len(year_months) < 2:
            print(f"[SKIP] Year {year} has <2 months; nothing to simulate.")
            continue
        print(f"\n==== Simulating YEAR {year} (portfolio resets here) ====")
        simulate_year(year_months)
