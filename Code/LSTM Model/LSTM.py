
from __future__ import annotations

import os, sys
import json
import pickle
import gc
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import requests
import pandas as pd
import json
from io import StringIO
import re, json, requests
from io import BytesIO

# =============================================================================
# Configuration
# =============================================================================
if getattr(sys, 'frozen', False):  # Running as a PyInstaller exe
    SCRIPT_DIR = os.path.dirname(sys.executable)
else:  # Running as plain .py
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


STARTING_YEAR_FILE = os.path.join(SCRIPT_DIR, "Starting_year.txt")

with open(STARTING_YEAR_FILE, "r") as f:
    line = f.read().strip()

SKIP_BEFORE_YEAR = int(line.split("=")[1].strip())


DATA_PATH = "https://drive.google.com/uc?export=download&id=11OIu2mowmZhncBKQiLZ7iE2lYwZ3uw6n"
FEE_PATH  = "https://drive.google.com/uc?export=download&id=1B_iFMbVWtG-C6wIgbVnEBCVyA5LonZsV"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "monthly_forecasts")
OUTPUT_DIR_YEARLY = os.path.join(SCRIPT_DIR, "monthly_forecasts_Yearly_portfolio")
WINDOW_SIZE = 1440  # Timesteps per sample (1 day of minutes)
HORIZONS = [1, 2, 5, 10, 30, 60, 300, 600, 1440, 2880, 7200, 10080]

# Keep the same dict name/keys for compatibility with previous scripts.
BASE_MODEL_PARAMS = dict(d_model=128, num_heads=4, num_layers=3, dropout=0.2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
torch.backends.cudnn.benchmark = True


# =============================================================================
# Diagnostic/Plotting Flags
# =============================================================================

PLOT_LOG_SPACE = False            # If True, plot mean errors in log-return space
EMA_ENABLE = True                 # If True, apply EMA smoothing to price forecasts
MASK_MEANS_WHERE_ACTUAL_EXISTS = False  # If True, compute means only where actuals exist
RECON_MODE = "direct"            # 'direct' uses pred log(fut/now); 'flipped' negates
BIAS_ENABLE = False               # If True, apply per-horizon bias correction (log-space)

SCALE_ENABLE = False              # If True, apply per-horizon scale correction (log-space)
SCALE_MIN = 0.3
SCALE_MAX = 3.0
SCALE_EPS = 1e-8

# =============================================================================
# Model Components
# =============================================================================

class LSTMMulti(nn.Module):
    """LSTM encoder with optional temporal average pooling and a linear head.

    Predicts a vector of multi-horizon targets from a window of past features.

    Args:
        input_size: Number of input features per timestep.
        output_size: Number of horizons predicted per sample.
        d_model: Hidden size of the LSTM (name retained for compatibility).
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout applied between LSTM layers (active if num_layers > 1).
        pool_k: If > 1, applies 1D average pooling with given kernel/stride to reduce T.
        num_heads: Unused; accepted for interface compatibility with Transformer configs.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        d_model: int = 128,
        num_heads: int = 4,  # ignored, for compatibility only
        num_layers: int = 3,
        dropout: float = 0.2,
        pool_k: int = 12,
    ) -> None:
        super().__init__()
        self.pool_k = pool_k
        self.input_ln = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [B, T, F].

        Returns:
            Tensor of shape [B, H] for H horizons.
        """
        x = self.input_ln(x)
        if self.pool_k and self.pool_k > 1:
            # Downsample over time for efficiency
            x = x.transpose(1, 2)
            x = nn.functional.avg_pool1d(x, kernel_size=self.pool_k, stride=self.pool_k)
            x = x.transpose(1, 2)
        out, _ = self.lstm(x)  # out: [B, T, d_model]
        return self.head(out[:, -1, :])  # Use the last timestep representation

# =============================================================================
# Data Loading
# =============================================================================

def _extract_gdrive_id(url_or_id: str) -> str:
    """Extract a Google Drive file ID from a URL or return the ID if provided.

    Raises:
        ValueError: If no valid file ID can be extracted.
    """
    # Accept a bare file id or any GDrive URL
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url_or_id)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url_or_id)
    if m:
        return m.group(1)
    # If it looks like an id already, use it
    if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", url_or_id):
        return url_or_id
    raise ValueError(f"Could not extract Google Drive file id from: {url_or_id}")

def _gdrive_download_bytes(url_or_id: str) -> bytes:
    """Download a Google Drive file, handling the virus-scan confirm page.

    Returns:
        Raw bytes of the downloaded file.

    Raises:
        RuntimeError: If Google Drive returns HTML instead of the file content.
    """
    file_id = _extract_gdrive_id(url_or_id)
    sess = requests.Session()

    uc = "https://drive.google.com/uc"
    r = sess.get(uc, params={"export": "download", "id": file_id}, allow_redirects=True)
    r.raise_for_status()

    ct = r.headers.get("Content-Type", "").lower()
    if "text/html" not in ct and "application/json" not in ct:
        return r.content

    text = r.text
    m = re.search(r'name="confirm"\s+value="([^"]+)"', text)
    token = m.group(1) if m else None

    dl = "https://drive.usercontent.google.com/download"
    params = {"id": file_id, "export": "download"}
    if token:
        params["confirm"] = token

    r2 = sess.get(dl, params=params, allow_redirects=True)
    r2.raise_for_status()

    # If still HTML, raise a helpful error
    ct2 = r2.headers.get("Content-Type", "").lower()
    if "text/html" in ct2 and "attachment" not in r2.headers.get("Content-Disposition", "").lower():
        raise RuntimeError("Google Drive returned an HTML page (likely still a confirm page). "
                           "Try again later or ensure the file is shared as 'Anyone with the link'.")
    return r2.content

def load_data() -> pd.DataFrame:
    """Load price and fee data, align by minute, and engineer base features.

    Returns:
        DataFrame indexed by naive UTC datetime with at least: Close, fee_usd,
        trading_fee_pct, and log_return.
    """
    print("Loading data...")

    # Use provided links (any valid GDrive URL or ID)
    csv_bytes = _gdrive_download_bytes(DATA_PATH)
    df = pd.read_csv(BytesIO(csv_bytes), low_memory=False)

    # Identify and parse timestamp column
    ts_candidates = ["Timestamp", "timestamp", "ts", "time", "unix", "Unix", "date", "Date", "datetime", "Datetime"]
    ts_col = next((c for c in ts_candidates if c in df.columns), None)
    if ts_col is None:
        raise KeyError(f"No timestamp-like column found. Columns: {list(df.columns)}")

    ts = df[ts_col]
    if pd.api.types.is_numeric_dtype(ts):
        med = float(pd.Series(ts).dropna().astype("float64").median())
        unit = "ms" if med > 1e12 else "s"
        dt = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        dt = pd.to_datetime(ts, utc=True, infer_datetime_format=True, errors="coerce")
        if dt.isna().all():
            raise ValueError(f"Failed to parse datetimes from '{ts_col}'")

    df.insert(0, "datetime", dt.dt.tz_convert(None))
    df.set_index("datetime", inplace=True)
    if ts_col in df.columns:  # keep OHLCV only
        df.drop(columns=[ts_col], inplace=True, errors="ignore")
    df = df.sort_index()

    # Normalize column names
    ren = {}
    for col in list(df.columns):
        low = str(col).strip().lower()
        if   low == "open":  ren[col] = "Open"
        elif low == "close": ren[col] = "Close"
        elif low == "high":  ren[col] = "High"
        elif low == "low":   ren[col] = "Low"
        elif low in ("volume", "volume_btc", "volumebtc", "volume (btc)"):
            ren[col] = "Volume"
    if ren:
        df.rename(columns=ren, inplace=True)
    if "Close" not in df.columns:
        raise KeyError(f"Missing 'Close' after rename. Columns: {list(df.columns)}")

    # Load fees JSON (robust to different shapes)
    fee_bytes = _gdrive_download_bytes(FEE_PATH)
    try:
        fees_json = json.loads(fee_bytes.decode("utf-8"))
    except Exception:
        # Some GDrive responses can be parsed directly by pandas
        fees_json = pd.read_json(BytesIO(fee_bytes)).to_dict(orient="list")

    # Expect {"fees-usd-per-transaction": [{"x": <ms>, "y": <usd>}, ...]}
    if isinstance(fees_json, dict) and "fees-usd-per-transaction" in fees_json:
        fees_df = pd.DataFrame(fees_json["fees-usd-per-transaction"])
    else:
        fees_df = pd.DataFrame(fees_json)

    x_col = next((c for c in fees_df.columns if str(c).lower() in ("x","timestamp","time","date","datetime","unix")), None)
    y_col = next((c for c in fees_df.columns if str(c).lower() in ("y","fee","fee_usd","fees","value")), None)
    if x_col is None or y_col is None:
        raise KeyError(f"Could not find time/value columns in fees JSON. Columns: {list(fees_df.columns)}")

    x = pd.Series(fees_df[x_col])
    if pd.api.types.is_numeric_dtype(x):
        med = float(x.dropna().astype("float64").median())
        unit = "ms" if med > 1e12 else "s"
        fees_dt = pd.to_datetime(x, unit=unit, utc=True)
    else:
        fees_dt = pd.to_datetime(x, utc=True, infer_datetime_format=True, errors="coerce")
    fees_series = (
        pd.Series(fees_df[y_col].values, index=fees_dt.dt.tz_convert(None))
          .rename("fee_usd")
          .resample("1min").interpolate()
    )

    # Merge and feature engineering
    df = df.join(fees_series, how="left").dropna(subset=["fee_usd"])
    df["trading_fee_pct"] = 0.001
    print("Data loaded and merged.")

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1)).fillna(0)

    return df


# =============================================================================
# Dataset
# =============================================================================

class BitcoinDataset(Dataset):
    """Sliding-window dataset mapping past features to multi-horizon targets."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, w: int) -> None:
        self.X = features.astype(np.float32)
        self.y = targets.astype(np.float32)
        self.w = w

    def __len__(self) -> int:
        return len(self.X) - self.w

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx : idx + self.w], self.y[idx + self.w]


# =============================================================================
# Training (with Early Stopping)
# =============================================================================

def train_model(
    df_train: pd.DataFrame,
    features: List[str],
    target_cols: List[str],
    model_path: str,
    scaler_path: str,
    label: str,
    max_epochs: int = 15,
    patience: int = 1,
    batch_size: int = 256,
    pool_k: int = 12,
):
    """Train an :class:`LSTMMulti` and persist weights and scalers.

    Early stopping is triggered on the plateau of training loss.

    Returns:
        Tuple of (trained model, feature scaler, target scaler).
    """
    print(f"Training model for {label} (max {max_epochs} epochs, patience {patience})...")
    df_t = df_train.copy()

    feat_scaler = StandardScaler()
    df_t.loc[:, features] = feat_scaler.fit_transform(df_t[features])
    targ_scaler = StandardScaler()
    df_t.loc[:, target_cols] = targ_scaler.fit_transform(df_t[target_cols])

    X_np = df_t[features].to_numpy(dtype=np.float32)
    y_np = df_t[target_cols].to_numpy(dtype=np.float32)

    ds = BitcoinDataset(X_np, y_np, WINDOW_SIZE)
    is_frozen = getattr(sys, "frozen", False)
    loader_kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    if is_frozen:
        loader_kwargs.update(num_workers=0)  # avoid worker spawn in frozen exe
    else:
        loader_kwargs.update(num_workers=4, persistent_workers=True, prefetch_factor=4)

    loader = DataLoader(**loader_kwargs)

    model = LSTMMulti(
        input_size=len(features),
        output_size=len(target_cols),
        **BASE_MODEL_PARAMS,
        pool_k=pool_k,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_loss = float("inf")
    no_improve = 0

    try:
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0
            model.train()
            batch_bar = tqdm(loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False)
            for i, (xb, yb) in enumerate(batch_bar, start=1):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
                batch_bar.set_postfix(loss=f"{total_loss / i:.6f}")

            avg_loss = total_loss / len(loader)
            print(f"  Epoch {epoch} - Avg Loss: {avg_loss:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Stopping after {epoch} epochs without improvement.")
                    break
    finally:
        # Ensure worker processes and caches are released
        del loader, ds, X_np, y_np
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump((feat_scaler, targ_scaler), f)
    print(f"Saved model+scalers for {label}\n")
    return model, feat_scaler, targ_scaler


# =============================================================================
# Calibration Helpers
# =============================================================================

@torch.no_grad()
def _predict_unscaled(
    df_slice: pd.DataFrame,
    model: nn.Module,
    feat_scaler: StandardScaler,
    targ_scaler: StandardScaler,
    features: List[str],
    target_cols: List[str],
    window_size: int,
) -> np.ndarray:
    """Produce **unscaled** log-return predictions with shape [N, H]."""
    X = feat_scaler.transform(df_slice[features])
    ds = BitcoinDataset(
        X.astype(np.float32),
        np.zeros((len(X), len(target_cols)), dtype=np.float32),
        window_size,
    )
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    preds = []
    model.eval()
    for xb, _ in loader:
        xb = xb.to(DEVICE)
        preds.append(model(xb).cpu().numpy())
    unp_log = targ_scaler.inverse_transform(np.vstack(preds))

    del loader, ds
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return unp_log


def compute_horizon_scale(
    df_slice: pd.DataFrame,
    model: nn.Module,
    feat_scaler: StandardScaler,
    targ_scaler: StandardScaler,
    features: List[str],
    target_cols: List[str],
    window_size: int = WINDOW_SIZE,
) -> np.ndarray:
    """Estimate per-horizon scale = std(true_log) / std(pred_log)."""
    unp_log = _predict_unscaled(df_slice, model, feat_scaler, targ_scaler, features, target_cols, window_size)

    start = window_size - 1
    stop = start + unp_log.shape[0]
    base = df_slice["Close"].iloc[start:stop].values

    horizons = [int(col.split("+")[1].replace("m", "")) for col in target_cols]
    scale = np.ones(len(horizons), dtype=np.float64)

    for j, h in enumerate(horizons):
        true_future = df_slice["Close"].shift(-h).iloc[start:stop].values
        true_log = np.log(true_future / base)
        pred_log = unp_log[:, j]
        m = ~np.isnan(true_log)
        if m.any():
            std_true = np.std(true_log[m])
            std_pred = np.std(pred_log[m])
            s = std_true / max(std_pred, SCALE_EPS)
            scale[j] = float(np.clip(s, SCALE_MIN, SCALE_MAX))
        else:
            scale[j] = 1.0

    return scale


def compute_horizon_bias(
    df_slice: pd.DataFrame,
    model: nn.Module,
    feat_scaler: StandardScaler,
    targ_scaler: StandardScaler,
    features: List[str],
    target_cols: List[str],
    window_size: int = WINDOW_SIZE,
) -> np.ndarray:
    """Estimate per-horizon bias = mean(pred_log - true_log)."""
    unp_log = _predict_unscaled(df_slice, model, feat_scaler, targ_scaler, features, target_cols, window_size)

    start = window_size - 1
    stop = start + unp_log.shape[0]
    base = df_slice["Close"].iloc[start:stop].values

    horizons = [int(col.split("+")[1].replace("m", "")) for col in target_cols]
    bias = np.zeros(len(horizons), dtype=np.float64)

    for j, h in enumerate(horizons):
        true_future = df_slice["Close"].shift(-h).iloc[start:stop].values
        true_log = np.log(true_future / base)
        pred_log = unp_log[:, j]
        m = ~np.isnan(true_log)
        bias[j] = (pred_log[m] - true_log[m]).mean() if m.any() else 0.0

    return bias


# =============================================================================
# Forecast Smoothing
# =============================================================================

def smooth_forecast(arr: np.ndarray, alpha: float = 0.5, beta: float = 0) -> np.ndarray:
    """Apply causal EMA with optional pull toward the causal running mean.

    Args:
        arr: Input series.
        alpha: EMA weight on the current point in [0, 1].
        beta:  Weight pulling EMA toward the causal running mean in [0, 1].

    Returns:
        Smoothed array of the same shape as ``arr``.
    """
    if arr.size == 0:
        return arr

    # Ensure float for stable math
    x = arr.astype(np.float64, copy=False)

    out = np.empty_like(x)
    out[0] = x[0]
    running_mean = x[0]

    for i in range(1, x.shape[0]):
        # Update mean of prefix [0..i]
        running_mean += (x[i] - running_mean) / (i + 1)

        # Causal EMA
        ema = alpha * x[i] + (1.0 - alpha) * out[i - 1]

        # Pull EMA toward causal running mean
        out[i] = ema + beta * (running_mean - ema)

    return out


# =============================================================================
# Month Forecasting
# =============================================================================

def forecast_month(
    df_period: pd.DataFrame,
    model: nn.Module,
    feat_scaler: StandardScaler,
    targ_scaler: StandardScaler,
    features: List[str],
    target_cols: List[str],
    label: str,
    pool_k: int = 12,
    bias: Optional[np.ndarray] = None,
    scale: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Run rolling forecasts across a single month window.

    Returns:
        DataFrame with forecasts, actuals, and auxiliary fields.
    """
    print(f"Forecasting for {label}...")
    df_p = df_period.copy()
    df_p.loc[:, features] = feat_scaler.transform(df_p[features])

    ds = BitcoinDataset(df_p[features].values, df_p[target_cols].values, WINDOW_SIZE)
    is_frozen = getattr(sys, "frozen", False)
    loader_kwargs = dict(dataset=ds, batch_size=1, shuffle=False, pin_memory=True)
    loader_kwargs.update(num_workers=0 if is_frozen else 0)  # keep 0 here; safe & simple
    loader = DataLoader(**loader_kwargs)

    preds_log = []
    with torch.no_grad():
        model.eval()
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            preds_log.append(model(xb).cpu().numpy().squeeze())

    del loader, ds
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    unp_log = targ_scaler.inverse_transform(np.vstack(preds_log))  # [N, H]

    if RECON_MODE == "flipped":
        unp_log = -unp_log

    if BIAS_ENABLE and (bias is not None):
        unp_log = unp_log - bias[None, :]

    if SCALE_ENABLE and (scale is not None):
        unp_log = unp_log * scale[None, :]

    last = df_period["Close"].iloc[WINDOW_SIZE - 1 : WINDOW_SIZE - 1 + len(unp_log)].values
    abs_pred = last[:, None] * np.exp(unp_log)

    if EMA_ENABLE:
        for j in range(abs_pred.shape[1]):
            abs_pred[:, j] = smooth_forecast(abs_pred[:, j])

    df_out = pd.DataFrame(abs_pred, columns=[f"forecast_t+{h}m" for h in HORIZONS])
    idx = df_period.index[WINDOW_SIZE - 1 : WINDOW_SIZE - 1 + len(unp_log)]
    df_out["datetime"] = idx
    df_out["close"] = df_period["Close"].iloc[WINDOW_SIZE - 1 : WINDOW_SIZE - 1 + len(unp_log)].values
    df_out["fee_usd"] = df_period["fee_usd"].iloc[WINDOW_SIZE - 1 : WINDOW_SIZE - 1 + len(unp_log)].values
    df_out["trading_fee_pct"] = df_period["trading_fee_pct"].iloc[WINDOW_SIZE - 1 : WINDOW_SIZE - 1 + len(unp_log)].values
    df_out["period"] = label

    for h in HORIZONS:
        df_out[f"actual_t+{h}m"] = (
            df_period["Close"].shift(-h).iloc[WINDOW_SIZE - 1 : WINDOW_SIZE - 1 + len(unp_log)].values
        )

    print(f"Forecast for {label} complete.\n")
    return df_out


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Prepare data and targets
    import multiprocessing as mp
    mp.freeze_support()  # <-- required for PyInstaller on Windows

    try:
        import torch.multiprocessing as tmp
        tmp.set_start_method("spawn", force=True)  # safe to call once
    except Exception:
        pass
    raw = load_data()
    df = raw.copy()
    for h in HORIZONS:
        df[f"target_t+{h}m"] = np.log(df["Close"].shift(-h) / df["Close"])
    df.dropna(inplace=True)

    features = ["Open", "Close", "log_return", "fee_usd"]
    target_cols = [f"target_t+{h}m" for h in HORIZONS]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR_YEARLY, exist_ok=True)
    periods = pd.period_range(df.index.min().to_period("M"), df.index.max().to_period("M"))
    if SKIP_BEFORE_YEAR is not None:
        periods = [p for p in periods if p.year >= SKIP_BEFORE_YEAR]
    metrics_csv = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    metrics_csv_Yearly = os.path.join(OUTPUT_DIR_YEARLY, "metrics_summary.csv")
    if not os.path.exists(metrics_csv):
        pd.DataFrame([], columns=["period", "MSE", "RMSE", "NRMSE", "MAPE (%)"]).to_csv(metrics_csv, index=False)
    if not os.path.exists(metrics_csv_Yearly):
        pd.DataFrame([], columns=["period", "MSE", "RMSE", "NRMSE", "MAPE (%)"]).to_csv(metrics_csv_Yearly, index=False)
    # Walk a 3-month training window forward, forecasting the month after it
    for i in range(len(periods) - 3):
        train_months = periods[i : i + 3]
        fore_month = periods[i + 3]
        csv_path = os.path.join(
            OUTPUT_DIR,
            str(fore_month.year),
            fore_month.strftime("%B_%Y"),
            f"forecast_{fore_month.strftime('%B_%Y')}.csv",
        )
        csv_path_Yearly = os.path.join(
            OUTPUT_DIR_YEARLY,
            str(fore_month.year),
            fore_month.strftime("%B_%Y"),
            f"forecast_{fore_month.strftime('%B_%Y')}.csv",
        )
        if os.path.exists(csv_path):
            train_months = periods[i : i + 3]
            label = (
                f"{train_months[0].strftime('%b %Y')}+{train_months[1].strftime('%b %Y')}+{train_months[2].strftime('%b %Y')} -> {fore_month.strftime('%b %Y')}"
            )
            print(f"Skipping {label}: forecast exists.")
            continue

        df_train = df[df.index.to_period("M").isin(train_months)]
        label = (
            f"{train_months[0].strftime('%b %Y')}+{train_months[1].strftime('%b %Y')}+{train_months[2].strftime('%b %Y')} -> {fore_month.strftime('%b %Y')}"
        )

        subdir = os.path.join(OUTPUT_DIR, str(fore_month.year), fore_month.strftime("%B_%Y"))
        os.makedirs(subdir, exist_ok=True)
        model_path = os.path.join(subdir, "model.pt")
        scaler_path = os.path.join(subdir, "scalers.pkl")
        csv_path = os.path.join(subdir, f"forecast_{fore_month.strftime('%B_%Y')}.csv")
        plot_path = os.path.join(subdir, f"performance_{fore_month.strftime('%B_%Y')}.png")

        subdir_yearly = os.path.join(OUTPUT_DIR_YEARLY, str(fore_month.year), fore_month.strftime("%B_%Y"))
        os.makedirs(subdir_yearly, exist_ok=True)
        model_path_yearly = os.path.join(subdir_yearly, "model.pt")
        scaler_path_yearly = os.path.join(subdir_yearly, "scalers.pkl")
        csv_path_Yearly = os.path.join(subdir_yearly, f"forecast_{fore_month.strftime('%B_%Y')}.csv")
        plot_path_yearly = os.path.join(subdir_yearly, f"performance_{fore_month.strftime('%B_%Y')}.png")

        # Load model and scalers if present; otherwise train from scratch
        if os.path.exists(model_path):
            model = LSTMMulti(
                input_size=len(features),
                output_size=len(target_cols),
                **BASE_MODEL_PARAMS,
                pool_k=12,
            ).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"Loaded model for {label} from {model_path}")

            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    feat_scaler, targ_scaler = pickle.load(f)
                print(f"Loaded scalers for {label} from {scaler_path}")
            else:
                print(f"Scalers missing for {label}, recomputing...")
                df_train = df[df.index.to_period("M").isin(train_months)]
                feat_scaler = StandardScaler().fit(df_train[features])
                targ_scaler = StandardScaler().fit(df_train[target_cols])
                with open(scaler_path, "wb") as f:
                    pickle.dump((feat_scaler, targ_scaler), f)
                print(f"Saved recomputed scalers for {label} to {scaler_path}")

            bias = None
            if BIAS_ENABLE:
                bias = compute_horizon_bias(
                    df_train, model, feat_scaler, targ_scaler, features, target_cols, WINDOW_SIZE
                )

            scale = None
            if SCALE_ENABLE:
                scale = compute_horizon_scale(
                    df_train, model, feat_scaler, targ_scaler, features, target_cols, WINDOW_SIZE
                )
        else:
            df_train = df[df.index.to_period("M").isin(train_months)]
            model, feat_scaler, targ_scaler = train_model(
                df_train,
                features,
                target_cols,
                model_path,
                scaler_path,
                label,
                max_epochs=15,
                patience=3,
                batch_size=256,
                pool_k=12,
            )

            bias = None
            if BIAS_ENABLE:
                bias = compute_horizon_bias(
                    df_train, model, feat_scaler, targ_scaler, features, target_cols, WINDOW_SIZE
                )

            scale = None
            if SCALE_ENABLE:
                scale = compute_horizon_scale(
                    df_train, model, feat_scaler, targ_scaler, features, target_cols, WINDOW_SIZE
                )

        # Forecast target month
        df_test = df[df.index.to_period("M") == fore_month]
        df_out = forecast_month(
            df_test,
            model,
            feat_scaler,
            targ_scaler,
            features,
            target_cols,
            label,
            pool_k=12,
            bias=bias,
            scale=scale,
        )

        # Persist outputs
        df_out.to_csv(csv_path, index=False)
        df_out.to_csv(csv_path_Yearly, index=False)
        print(f"Saved forecast for {label} to {csv_path}")

        # Aggregated horizon-wise performance plot for the month
        plt.figure()
        mins = HORIZONS

        if PLOT_LOG_SPACE:
            mean_true, mean_pred = [], []
            for h in HORIZONS:
                a_log = np.log(df_out[f"actual_t+{h}m"] / df_out["close"])
                p_log = np.log(df_out[f"forecast_t+{h}m"] / df_out["close"])
                if MASK_MEANS_WHERE_ACTUAL_EXISTS:
                    m = ~np.isnan(a_log)
                    mean_true.append(a_log[m].mean())
                    mean_pred.append(p_log[m].mean())
                else:
                    mean_true.append(a_log.mean())
                    mean_pred.append(p_log.mean())
            plt.plot(mins, mean_true, label="Mean True (log)")
            plt.plot(mins, mean_pred, label="Mean Predicted (log)")
            plt.ylabel("Log return")
        else:
            mean_true, mean_pred = [], []
            for h in HORIZONS:
                a = df_out[f"actual_t+{h}m"]
                p = df_out[f"forecast_t+{h}m"]
                if MASK_MEANS_WHERE_ACTUAL_EXISTS:
                    m = ~a.isna()
                    mean_true.append(a[m].mean())
                    mean_pred.append(p[m].mean())
                else:
                    mean_true.append(a.mean())
                    mean_pred.append(p.mean())
            plt.plot(mins, mean_true, label="Mean True")
            plt.plot(mins, mean_pred, label="Mean Predicted")
            plt.ylabel("Price")

        plt.title(f"Average Forecast vs True ({label})")
        plt.xlabel("Horizon (minutes)")
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.savefig(plot_path_yearly, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved performance plot for {label} to {plot_path}")

        # Monthly metrics across all horizons/timesteps
        P = df_out[[f"forecast_t+{h}m" for h in HORIZONS]].values.ravel()
        A = df_out[[f"actual_t+{h}m" for h in HORIZONS]].values.ravel()
        mask = (~np.isnan(P)) & (~np.isnan(A))
        Pv, Av = P[mask], A[mask]
        mse = np.mean((Pv - Av) ** 2) if len(Pv) > 0 else float("nan")
        rmse = np.sqrt(mse) if not np.isnan(mse) else float("nan")
        mean_c = df_out["close"].mean()
        nrmse = rmse / mean_c if mean_c else float("nan")
        mape = np.mean(np.abs((Pv - Av) / Av)) * 100 if np.all(Av != 0) else float("nan")
        print(f"  MSE: {mse:.6f}, RMSE: {rmse:.6f}, NRMSE: {nrmse:.6f}, MAPE: {mape:.2f}%\n")

        pd.DataFrame(
            [
                {
                    "period": fore_month.strftime("%B %Y"),
                    "MSE": mse,
                    "RMSE": rmse,
                    "NRMSE": nrmse,
                    "MAPE (%)": mape,
                }
            ]
        ).to_csv(metrics_csv, mode="a", index=False, header=False)
        pd.DataFrame(
            [
                {
                    "period": fore_month.strftime("%B %Y"),
                    "MSE": mse,
                    "RMSE": rmse,
                    "NRMSE": nrmse,
                    "MAPE (%)": mape,
                }
            ]
        ).to_csv(metrics_csv_Yearly, mode="a", index=False, header=False)

    # Append overall averages across all forecasted months
    mdf = pd.read_csv(metrics_csv)
    o_mse, o_rmse, o_nrmse, o_mape = (
        mdf["MSE"].mean(),
        mdf["RMSE"].mean(),
        mdf["NRMSE"].mean(),
        mdf["MAPE (%)"].mean(),
    )
    print(
        f"Overall avg MSE: {o_mse:.6f}, RMSE: {o_rmse:.6f}, NRMSE: {o_nrmse:.6f}, MAPE: {o_mape:.2f}%"
    )
    pd.DataFrame(
        [
            {
                "period": "Overall",
                "MSE": o_mse,
                "RMSE": o_rmse,
                "NRMSE": o_nrmse,
                "MAPE (%)": o_mape,
            }
        ]
    ).to_csv(metrics_csv, mode="a", index=False, header=False)
    print("Metrics summary updated.")
    pd.DataFrame(
        [
            {
                "period": "Overall",
                "MSE": o_mse,
                "RMSE": o_rmse,
                "NRMSE": o_nrmse,
                "MAPE (%)": o_mape,
            }
        ]
    ).to_csv(metrics_csv_Yearly, mode="a", index=False, header=False)
    print("Metrics summary updated.")
