"""
10. TimesFM 2.5 - Zero-Shot Forecasting (univariate)

Model family: TimesFM 2.5
Adaptation: Zero-Shot Forecasting
Information set: VIX univariate only
Colab layout: run this complete file as one cell on a T4 GPU
Fixed windows: 16, 32, 64, 128, 256. No window selection is performed.
Fixed horizons: 1, 5, 22 trading days.
Model ID: google/timesfm-2.5-200m-transformers
Checkpoint revision: 5a9806b9b291fad9233b5249d88263f1846304d3
Forecast configuration: direct 22-step log-VIX forecast, q10/q50/q90, q50 point forecast.
Evaluation configuration: level MAE primary, HAR-OLS benchmark, non-overlapping offset 0 primary.
Output configuration: one compact 15-row table is displayed and saved unchanged to this cell's CSV.
HAR note: Corsi's daily/weekly/monthly structure is estimated by OLS on log-VIX with direct horizons.
Update configuration: No parameter updates. The pretrained pipeline is loaded once and reused for every window.
Dependency policy: the Colab NumPy, pandas, SciPy, statsmodels, scikit-learn and PyTorch stack is preserved.
Live data policy: disabled. An approved VIX CSV is mandatory.

Documentation:
  https://github.com/google-research/timesfm
  https://huggingface.co/google/timesfm-2.5-200m-transformers
  https://huggingface.co/google/timesfm-2.5-200m-transformers/commit/5a9806b9b291fad9233b5249d88263f1846304d3
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/timesfm2_5/modular_timesfm2_5.py

Installation notes:
  pip install "transformers==5.3.0"

Part of the VIX foundation-model forecasting benchmark.
Run as a standalone script on a CUDA GPU (tested on a T4).
"""

# Imports
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data_io import REPO_DATA_DIR, locate_approved_csv, resolve_results_directory
from common.plotting import enable_figure_autosave
from common.evaluation import mean_absolute_error
import importlib.metadata
import math
import platform
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from IPython.display import display
from scipy.stats import t as student_t
from transformers import TimesFm2_5ModelForPrediction

# Configuration
MODEL_FAMILY = "timesfm_2_5"
MODEL_ID = "google/timesfm-2.5-200m-transformers"
MODEL_REVISION = "5a9806b9b291fad9233b5249d88263f1846304d3"
WINDOW_SIZES = [16, 32, 64, 128, 256]
HORIZONS = [1, 5, 22]
HORIZON_STEPS = [0, 4, 21]
MAX_HORIZON = 22
HAR_WEEK = 5
HAR_MONTH = 22
QUANTILE_INDICES = [1, 5, 9]
INFERENCE_BATCH_SIZE = 64
SEED = 2026
BOOTSTRAP_REPLICATIONS = 5000
BOOTSTRAP_SEED = 2026
OUTPUT_FILENAME = "forecast_timesfm_2_5_zero_shot_univariate.csv"
LOCAL_VIX_PATHS = [REPO_DATA_DIR / "VIX.csv", Path("/content/VIX.csv"), Path("/content/VIXCLS.csv"), Path("/content/vix.csv")]
RESULTS_DIRECTORY = resolve_results_directory()
enable_figure_autosave(RESULTS_DIRECTORY, Path(OUTPUT_FILENAME).stem)

if not torch.cuda.is_available():
    raise RuntimeError("Select a T4 GPU runtime in Google Colab before running this cell.")

DEVICE = torch.device("cuda")
MODEL_DTYPE = torch.float16
print(
    f"Python {platform.python_version()} | PyTorch {torch.__version__} | "
    f"transformers {importlib.metadata.version('transformers')} | GPU {torch.cuda.get_device_name(0)}"
)


# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Data




def load_vix_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    schemas = [("Date", "Close"), ("DATE", "VIXCLS")]
    schema = next((columns for columns in schemas if set(columns).issubset(raw.columns)), None)
    if schema is None:
        raise ValueError("Accepted schemas are exactly Date,Close or DATE,VIXCLS.")

    date_column, value_column = schema
    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(raw[date_column], errors="coerce"),
            "VIX": pd.to_numeric(raw[value_column], errors="coerce"),
        }
    )
    valid_rows = data["Date"].notna() & data["VIX"].notna() & data["VIX"].gt(0)
    dropped_rows = int((~valid_rows).sum())
    data = data.loc[valid_rows].sort_values("Date").reset_index(drop=True)

    if data.empty:
        raise ValueError("The approved CSV contains no valid positive VIX observations.")
    if data["Date"].duplicated().any():
        duplicate_dates = data.loc[data["Date"].duplicated(keep=False), "Date"]
        raise ValueError(f"Duplicate VIX dates found: {duplicate_dates.head().tolist()}")

    print(f"VIX source: {path} | dropped rows: {dropped_rows}")
    print(f"Range: {data['Date'].iloc[0].date()} to {data['Date'].iloc[-1].date()} | n={len(data)}")
    return data


vix_data = load_vix_csv(locate_approved_csv(LOCAL_VIX_PATHS, "VIX"))
dates = vix_data["Date"].to_numpy()
vix_level = vix_data["VIX"].to_numpy(dtype=np.float64)
log_vix = np.log(vix_level)


# Split
test_start = int(np.floor(0.80 * len(vix_data)))
minimum_origin = max(max(WINDOW_SIZES) - 1, HAR_MONTH - 1)
test_origins = np.arange(max(test_start, minimum_origin), len(vix_data) - MAX_HORIZON, dtype=int)
if len(test_origins) == 0:
    raise ValueError("The cleaned VIX history is too short for the fixed 80% test split.")
print(f"Test origins: {len(test_origins)}")


# Contexts
def build_contexts(values: np.ndarray, origins: np.ndarray, window_size: int) -> np.ndarray:
    offsets = np.arange(-window_size + 1, 1)
    return values[origins[:, None] + offsets].astype(np.float32, copy=False)


def to_level(log_forecasts: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(log_forecasts, dtype=np.float64), -20.0, 20.0)
    clip_count = int(np.count_nonzero(clipped != log_forecasts))
    if clip_count:
        warnings.warn(
            f"Clipped {clip_count} log forecasts to [-20, 20] before exponentiation.",
            RuntimeWarning,
        )
    return np.exp(clipped)


# HAR benchmark
def har_features(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    return np.column_stack(
        [
            np.ones(len(values)),
            values,
            series.rolling(HAR_WEEK, min_periods=HAR_WEEK).mean(),
            series.rolling(HAR_MONTH, min_periods=HAR_MONTH).mean(),
        ]
    )


def expanding_har_ols(values: np.ndarray, origins: np.ndarray) -> dict[int, np.ndarray]:
    features = har_features(values)
    forecasts: dict[int, np.ndarray] = {}

    for horizon in HORIZONS:
        log_forecasts = np.empty(len(origins), dtype=np.float64)
        for position, origin in enumerate(origins):
            training_origins = np.arange(HAR_MONTH - 1, origin - horizon + 1)
            parameters = sm.OLS(
                values[training_origins + horizon],
                features[training_origins],
            ).fit().params
            log_forecasts[position] = features[origin] @ parameters
        forecasts[horizon] = to_level(log_forecasts)

    return forecasts


# Statistical comparison


def diebold_mariano_p_value(loss_differential: np.ndarray) -> float:
    differential = np.asarray(loss_differential, dtype=np.float64)
    n_obs = len(differential)
    if n_obs < 3:
        return np.nan

    lag = min(int(np.floor(n_obs ** (1 / 3))), n_obs - 1)
    centered = differential - differential.mean()
    long_run_variance = float(centered @ centered / n_obs)
    for current_lag in range(1, lag + 1):
        covariance = float(centered[current_lag:] @ centered[:-current_lag] / n_obs)
        long_run_variance += 2.0 * (1.0 - current_lag / (lag + 1.0)) * covariance

    if long_run_variance <= np.finfo(float).eps:
        return np.nan

    statistic = differential.mean() / math.sqrt(long_run_variance / n_obs)
    statistic *= math.sqrt((n_obs - 1.0) / n_obs)
    return float(2.0 * student_t.sf(abs(statistic), df=n_obs - 1))


def moving_block_bootstrap_ci(loss_differential: np.ndarray) -> tuple[float, float]:
    differential = np.asarray(loss_differential, dtype=np.float64)
    n_obs = len(differential)
    if n_obs < 2:
        return np.nan, np.nan

    block_length = min(max(2, int(round(n_obs ** (1 / 3)))), n_obs)
    starts = np.arange(n_obs - block_length + 1)
    blocks_needed = math.ceil(n_obs / block_length)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_REPLICATIONS, dtype=np.float64)

    for replication in range(BOOTSTRAP_REPLICATIONS):
        sampled_starts = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate(
            [differential[start : start + block_length] for start in sampled_starts]
        )[:n_obs]
        means[replication] = sample.mean()

    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


# Model
model = TimesFm2_5ModelForPrediction.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    torch_dtype=MODEL_DTYPE,
).to(DEVICE)
model.eval()
patch_length = int(model.config.patch_length)
if patch_length <= 0:
    raise ValueError(f"Invalid TimesFM patch length: {patch_length}")
print(f"Loaded {MODEL_ID} at {MODEL_REVISION} | patch length={patch_length}")


def predict_log_quantiles(contexts: np.ndarray) -> np.ndarray:
    padded_context_length = math.ceil(contexts.shape[1] / patch_length) * patch_length
    batches: list[np.ndarray] = []
    start = 0
    batch_size = min(INFERENCE_BATCH_SIZE, len(contexts))

    while start < len(contexts):
        stop = min(start + batch_size, len(contexts))
        past_values = [
            torch.as_tensor(row, dtype=torch.float32, device=DEVICE)
            for row in contexts[start:stop]
        ]
        try:
            with torch.inference_mode():
                output = model(
                    past_values=past_values,
                    forecast_context_len=padded_context_length,
                    truncate_negative=False,
                )
            full_predictions = output.full_predictions.detach().float().cpu().numpy()
            invalid_shape = (
                full_predictions.ndim != 3
                or full_predictions.shape[1] < MAX_HORIZON
                or full_predictions.shape[2] != 10
            )
            if invalid_shape:
                raise ValueError(f"Unexpected full_predictions shape: {full_predictions.shape}")
            batches.append(full_predictions[:, :MAX_HORIZON, QUANTILE_INDICES])
            start = stop
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            warnings.warn(f"CUDA OOM: inference batch reduced to {batch_size}.", RuntimeWarning)

    quantiles = np.concatenate(batches)
    expected_shape = (len(contexts), MAX_HORIZON, len(QUANTILE_INDICES))
    if quantiles.shape != expected_shape or not np.isfinite(quantiles).all():
        raise ValueError(f"Invalid quantile forecasts: shape={quantiles.shape}")
    if np.any(np.diff(quantiles, axis=-1) < 0):
        quantiles = np.sort(quantiles, axis=-1)
    return quantiles


# Forecasts and metrics
test_actual = np.column_stack([vix_level[test_origins + horizon] for horizon in HORIZONS])
test_har = expanding_har_ols(log_vix, test_origins)
forecast_store: dict[int, np.ndarray] = {}
result_rows: list[dict[str, object]] = []

for window_size in WINDOW_SIZES:
    print(f"Forecasting W={window_size}")
    contexts = build_contexts(log_vix, test_origins, window_size)
    log_quantiles = predict_log_quantiles(contexts)[:, HORIZON_STEPS, :]
    level_quantiles = to_level(log_quantiles)
    forecast_store[window_size] = level_quantiles

    for horizon_index, horizon in enumerate(HORIZONS):
        positions = np.arange(0, len(test_origins), horizon)
        actual = test_actual[positions, horizon_index]
        model_forecast = level_quantiles[positions, horizon_index, 1]
        har_forecast = test_har[horizon][positions]
        model_mae = mean_absolute_error(actual, model_forecast)
        har_mae = mean_absolute_error(actual, har_forecast)
        loss_differential = np.abs(actual - model_forecast) - np.abs(actual - har_forecast)
        bootstrap_lower, bootstrap_upper = moving_block_bootstrap_ci(loss_differential)

        result_rows.append(
            {
                "window_size": window_size,
                "horizon": horizon,
                "benchmark": "HAR-OLS",
                "n_obs": len(actual),
                "model_MAE": model_mae,
                "har_MAE": har_mae,
                "delta_MAE": model_mae - har_mae,
                "dm_p_value_unadjusted": diebold_mariano_p_value(loss_differential),
                "bootstrap_ci_lower": bootstrap_lower,
                "bootstrap_ci_upper": bootstrap_upper,
            }
        )


# CSV
OUTPUT_COLUMNS = [
    "window_size",
    "horizon",
    "benchmark",
    "n_obs",
    "model_MAE",
    "har_MAE",
    "delta_MAE",
    "dm_p_value_unadjusted",
    "bootstrap_ci_lower",
    "bootstrap_ci_upper",
]
results = (
    pd.DataFrame(result_rows, columns=OUTPUT_COLUMNS)
    .sort_values(["window_size", "horizon"])
    .reset_index(drop=True)
)

RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
output_path = RESULTS_DIRECTORY / OUTPUT_FILENAME
results.to_csv(output_path, index=False)
print(f"Saved compact results CSV: {output_path}")


# Display
print("Primary test comparison, identical to the saved CSV")
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 180):
    display(results)

primary_har_ols = results


# Plots
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for horizon_index, horizon in enumerate(HORIZONS):
    horizon_rows = primary_har_ols[
        primary_har_ols["horizon"] == horizon
    ].sort_values("window_size")
    axes[horizon_index].plot(horizon_rows["window_size"], horizon_rows["model_MAE"], marker="o", label=MODEL_FAMILY)
    axes[horizon_index].plot(horizon_rows["window_size"], horizon_rows["har_MAE"], marker="s", label="HAR-OLS")
    axes[horizon_index].set_title(f"MAE, h={horizon}")
    axes[horizon_index].set_xlabel("Window")
    axes[horizon_index].grid(alpha=0.3)
axes[0].legend()
fig.tight_layout()
plt.show()
plt.close(fig)

delta_matrix = primary_har_ols.pivot(index="window_size", columns="horizon", values="delta_MAE").reindex(index=WINDOW_SIZES, columns=HORIZONS)
fig, ax = plt.subplots(figsize=(6, 5))
image = ax.imshow(delta_matrix.to_numpy(), cmap="RdBu_r", aspect="auto")
ax.set_xticks(range(len(HORIZONS)), HORIZONS)
ax.set_yticks(range(len(WINDOW_SIZES)), WINDOW_SIZES)
ax.set_xlabel("Horizon")
ax.set_ylabel("Window")
ax.set_title("MAE difference vs HAR-OLS")
fig.colorbar(image, ax=ax, label="Model MAE - HAR MAE")
fig.tight_layout()
plt.show()
plt.close(fig)

for horizon_index, horizon in enumerate(HORIZONS):
    fig, axes = plt.subplots(len(WINDOW_SIZES), 1, figsize=(14, 13), sharex=True)
    plot_positions = np.arange(0, len(test_origins), horizon)
    target_dates = pd.to_datetime(dates[test_origins[plot_positions] + horizon])
    observed = test_actual[plot_positions, horizon_index]
    for axis, window_size in zip(axes, WINDOW_SIZES):
        quantiles = forecast_store[window_size][plot_positions, horizon_index]
        axis.plot(target_dates, observed, color="black", linewidth=1, label="Observed")
        axis.plot(target_dates, quantiles[:, 1], color="tab:blue", linewidth=1, label="q50")
        axis.plot(
            target_dates,
            test_har[horizon][plot_positions],
            color="tab:orange",
            linewidth=1,
            label="HAR-OLS",
        )
        axis.fill_between(target_dates, quantiles[:, 0], quantiles[:, 2], color="tab:blue", alpha=0.18, label="q10-q90")
        axis.set_title(f"W={window_size}")
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=4)
    fig.suptitle(f"Non-overlapping test paths, horizon {horizon}")
    fig.tight_layout()
    plt.show()
    plt.close(fig)

"""### Fine Tuned

"""

# Commented out IPython magic to ensure Python compatibility.

# Commented out IPython magic to ensure Python compatibility.
# %pip install -q --upgrade-strategy only-if-needed "transformers==5.3.0" "peft==0.18.1"

